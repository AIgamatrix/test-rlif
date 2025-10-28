from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# 说明：
# 本文件使用 Transformers 对生成序列执行“逐步打分”，以获得完整词表logits。
# 之所以保留该步骤，是因为熵与 D_KL(U || p) 的精确计算需要完整词表分布。


class HFScorer:
    """Transformers 打分封装类（支持多 GPU 通过 device_map=auto）。"""

    def __init__(
        self,
        model_path: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token is None:
            # 某些因果语言模型无 pad_token，设置为 eos 以便 batch 处理
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    def score_next_token_logits(
        self,
        prompts: List[str],
        generations: List[str],
        max_length: int = 4096,
    ) -> List[Dict[str, Any]]:
        """
        对每个 (prompt, generation) 执行一次性前向，提取对已生成序列每一步的下一令牌logits。

        返回每个样本的：
        - logits_seq: 张量形状 [T, V]（T为生成序列长度，V为词表大小）
        - target_ids: 张量形状 [T]（生成文本的token id序列，与 logits_seq 对齐）
        - context_len: 上下文（prompt）长度，便于定位起始时间步
        """
        results: List[Dict[str, Any]] = []
        for prompt, gen in zip(prompts, generations):
            # 编码输入：prompt + generation
            enc = self.tokenizer(
                prompt,
                add_special_tokens=True,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            prompt_ids = enc["input_ids"]  # [1, Lp]

            gen_enc = self.tokenizer(
                gen,
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            gen_ids = gen_enc["input_ids"]  # [1, Tg]

            # 合并：模型输入为 [prompt_ids, gen_ids]
            input_ids = torch.cat([prompt_ids, gen_ids], dim=1)  # [1, Lp+Tg]
            attn_mask = torch.ones_like(input_ids)

            # 将输入移动到模型设备的第一个可用设备（device_map=auto下可能是分片）
            input_ids = input_ids.to(self.model.device)
            attn_mask = attn_mask.to(self.model.device)

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
                logits = outputs.logits  # [1, Lp+Tg, V]

            # 我们需要与生成序列对齐的下一令牌 logits：
            # 位置 k 的logits用于预测 token at k（即下一令牌），因此
            # 对生成的第 t 步（0-based），对应的logits位置是 (Lp - 1 + t)
            Lp = prompt_ids.size(1)
            Tg = gen_ids.size(1)
            # 切片取生成段的下一令牌预测logits
            # 注意：logits 的 i 位置预测的是 input_ids[i+1]
            # 因此对目标令牌序列 gen_ids[0..Tg-1]，应取 logits[ (Lp-1) .. (Lp-1+Tg-1) ]
            start = max(Lp - 1, 0)
            end = start + Tg
            logits_seq = logits[:, start:end, :].squeeze(0).detach().cpu()  # [Tg, V]
            target_ids = gen_ids.squeeze(0).detach().cpu()                  # [Tg]

            results.append({
                "logits_seq": logits_seq,
                "target_ids": target_ids,
                "context_len": int(Lp),
            })
        return results

    def close(self) -> None:
        """释放 HF 模型与显存，避免逐模型循环时 OOM。"""
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass