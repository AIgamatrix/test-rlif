from typing import List, Dict, Any
import gc
import torch
from vllm import LLM, SamplingParams


# 说明：
# 本文件使用 vLLM 进行多 GPU 高效推理，输出文本与选定令牌的logprob。
# 后续将使用 Transformers 对生成序列进行逐步打分以获取完整词表分布。


class VLLMGenerator:
    """vLLM 推理封装类。"""

    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.7,
        enforce_eager: bool = True, 
    ) -> None:
        # 初始化 vLLM 实例
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )

    def generate(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        top_k: int = -1,
    ) -> List[Dict[str, Any]]:
        """
        使用 vLLM 执行批量生成。

        返回每个样本的：
        - text: 生成的文本
        - token_ids: 生成的 token id 序列
        - logprobs: 每步选定令牌的 logprob（来自 vLLM 的 top-logprobs，其中包含选定令牌）
        """
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            n=1,
            stop=None,
            logprobs=True,  # 返回每步的top-logprobs（包含选定令牌）
        )

        outputs = self.llm.generate(prompts, sampling_params)
        results: List[Dict[str, Any]] = []
        for out in outputs:
            # vLLM 可能返回多条完成，此处取第一条
            comp = out.outputs[0]
            # vLLM 结构：comp.token_ids, comp.text, comp.logprobs（列表，元素为字典）
            # 每步 logprobs 中，键为 token 字符串或特殊键，值为 logprob。
            # 选定令牌的 logprob 可通过 comp.logprobs[t].get(comp.tokens[t]) 获得，但 tokens 可能是字符串。
            # 保险起见，直接保存 vLLM 给出的 token_ids 与 logprobs 列表，后续由 transformers 重建 token 对齐。
            results.append({
                "text": comp.text,
                "token_ids": comp.token_ids,
                "logprobs": comp.logprobs,
            })
        return results

    def close(self) -> None:
        """释放 vLLM 的缓存与显存，占用较多时建议每个模型用完即调。"""
        try:
            # 某些版本的 vLLM 暴露 free_cache_engine 接口
            if hasattr(self.llm, "free_cache_engine"):
                self.llm.free_cache_engine()
        except Exception:
            pass
        try:
            # 主动删除对象，随后清理缓存
            del self.llm
        except Exception:
            pass
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def __del__(self):
        # 兜底释放
        try:
            self.close()
        except Exception:
            pass