#!/usr/bin/env python3
"""
基于 llm_entropy.ipynb 的逻辑，提供后台批处理脚本：
- 逐模型顺序加载 -> 逐题生成 -> 计算熵曲线与平均对数概率
- 统一释放 vLLM/HF 资源，降低显存占用，避免 OOM
- 生成与 notebook 类似的按模型聚合的绘图（正确/错误各一张）
- 保存 JSON/CSV + 图像到指定目录

用法示例：
nohup python entropy_metrics/run_amc12_batch.py \
  --model-paths /home/llama/test-rlif/Qwen2.5-Math-1.5B,/home/llama/test-rlif/Qwen2.5-0.5B-Instruct \
  --data-path /home/llama/test-rlif/datasets/amc12/data/train-00000-of-00001.parquet \
  --num-samples 50 \
  --max-new-tokens 512 \
  --gpu-mem-util 0.6 \
  --output-dir /home/llama/test-rlif/entropy_metrics/outputs_batch \
  > /home/llama/test-rlif/entropy_metrics/outputs_batch/run.log 2>&1 &

依赖：transformers, torch, pandas, matplotlib, vllm
"""

import os
import re
import gc
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

# 复用熵度量实现
try:
    from entropy_metrics.metrics import token_entropy
except Exception:
    from metrics import token_entropy

# 集成现有推理/打分封装
try:
    from entropy_metrics.infer_vllm import VLLMGenerator
    from entropy_metrics.infer_transformers import HFScorer
except Exception:
    from infer_vllm import VLLMGenerator
    from infer_transformers import HFScorer

# matplotlib 设置，确保中文与负号正常显示
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class ModelBundle:
    path: str
    tokenizer: Any
    vllm: VLLMGenerator
    scorer: HFScorer


def load_model(model_path: str, tp_size: int = 1, max_model_len: int = 4096, gpu_mem_utilization: float = 0.7) -> ModelBundle:
    from transformers import AutoTokenizer, AutoConfig
    # tokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # vLLM 生成器
    vllm = VLLMGenerator(
        model_path=model_path,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_utilization,
    )
    # HF 打分器（用于下一 token logits）
    scorer = HFScorer(model_path=model_path)
    return ModelBundle(path=model_path, tokenizer=tok, vllm=vllm, scorer=scorer)


def load_amc12_df(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"AMC12数据集未找到: {data_path}")
    df = pd.read_parquet(data_path)
    if not set(["id", "problem", "answer"]).issubset(df.columns):
        raise ValueError("AMC12数据集缺少必要列: id/problem/answer")
    return df


def format_prompt(problem: str) -> str:
    return (
        "请解决以下数学问题，并给出最终答案。请仔细思考，然后在末尾提供最终答案。\n"
        "请将最终答案包裹在\\boxed{}中。\n"
        f"问题：{problem}\n"
    )


def apply_chat_template(bundle: ModelBundle, prompt_text: str) -> str:
    tok = bundle.tokenizer
    if hasattr(tok, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt_text}]
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return prompt_text
    return prompt_text


def extract_answer(response: str) -> str:
    m = re.search(r"\\boxed\{(.+?)\}", response)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", response)
    return nums[-1].strip() if nums else ""


def to_float_or_none(s: str):
    try:
        return float(s)
    except Exception:
        return None


def evaluate_correctness(correct_answer: float, extracted: str) -> Tuple[bool, float]:
    pred = to_float_or_none(extracted)
    if pred is None:
        return False, float("nan")
    try:
        is_ok = float(correct_answer) == float(pred)
    except Exception:
        is_ok = False
    return is_ok, float(pred)


def generate_with_scores(bundle: ModelBundle, prompt_text: str, max_new_tokens: int, temperature: float, top_p: float, top_k: int) -> Dict[str, Any]:
    final_prompt = apply_chat_template(bundle, prompt_text)
    vout_list = bundle.vllm.generate(
        prompts=[final_prompt],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    vout = vout_list[0]
    response_text = vout.get("text", "")

    scored = bundle.scorer.score_next_token_logits(prompts=[final_prompt], generations=[response_text])
    if not scored:
        entropy_series = []
        avg_logprob = None
    else:
        item = scored[0]
        logits_seq = item["logits_seq"]
        target_ids = item["target_ids"]
        ent = token_entropy(logits_seq)
        entropy_series = [float(x) for x in ent]
        logp = torch.log_softmax(logits_seq, dim=-1)
        tgt_logp = logp.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        avg_logprob = float(tgt_logp.mean()) if tgt_logp.numel() > 0 else None

    return {
        "response_text": response_text,
        "entropy_series": entropy_series,
        "avg_logprob": avg_logprob,
    }


def plot_entropy_curves_per_model(df: pd.DataFrame, save_dir: str, font_prop: FontProperties | None = None):
    os.makedirs(save_dir, exist_ok=True)
    for model_name, group in df.groupby("model_name"):
        correct_samples = group[group["is_correct"] == True]
        incorrect_samples = group[group["is_correct"] == False]

        # 正确样例曲线
        if len(correct_samples) > 0:
            plt.figure(figsize=(9, 5))
            for _, row in correct_samples.iterrows():
                series = row["entropy_series"]
                if not series:
                    continue
                steps = list(range(1, len(series) + 1))
                plt.plot(steps, series, label=f"Q{row['id']}")
            ax = plt.gca()
            if font_prop is not None:
                plt.title(f"{model_name} - 答案正确的熵变曲线", fontproperties=font_prop)
                plt.xlabel("生成步骤", fontproperties=font_prop)
                plt.ylabel("熵值", fontproperties=font_prop)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    try:
                        label.set_fontproperties(font_prop)
                    except Exception:
                        pass
                leg = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop=font_prop)
            else:
                plt.title(f"{model_name} - 答案正确的熵变曲线")
                plt.xlabel("生成步骤")
                plt.ylabel("熵值")
                leg = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            if len(correct_samples) > 0:
                xmax = max(len(s) for s in correct_samples["entropy_series"] if s)
                plt.xlim(1, max(1, xmax))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(save_dir, f"{model_name}_correct_entropy.png")
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()

        # 错误样例曲线
        if len(incorrect_samples) > 0:
            plt.figure(figsize=(9, 5))
            for _, row in incorrect_samples.iterrows():
                series = row["entropy_series"]
                if not series:
                    continue
                steps = list(range(1, len(series) + 1))
                plt.plot(steps, series, label=f"Q{row['id']}")
            ax = plt.gca()
            if font_prop is not None:
                plt.title(f"{model_name} - 答案错误的熵变曲线", fontproperties=font_prop)
                plt.xlabel("生成步骤", fontproperties=font_prop)
                plt.ylabel("熵值", fontproperties=font_prop)
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    try:
                        label.set_fontproperties(font_prop)
                    except Exception:
                        pass
                leg = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", prop=font_prop)
            else:
                plt.title(f"{model_name} - 答案错误的熵变曲线")
                plt.xlabel("生成步骤")
                plt.ylabel("熵值")
                leg = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            if len(incorrect_samples) > 0:
                xmax = max(len(s) for s in incorrect_samples["entropy_series"] if s)
                plt.xlim(1, max(1, xmax))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(alpha=0.3)
            plt.tight_layout()
            out_png = os.path.join(save_dir, f"{model_name}_incorrect_entropy.png")
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()


def run(args):
    df = load_amc12_df(args.data_path)
    if args.num_samples > 0:
        df_sample = df.iloc[:args.num_samples, :].copy()
    else:
        df_sample = df

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots_per_model")
    os.makedirs(plots_dir, exist_ok=True)

    # 字体
    font_prop = None
    if args.font_path and os.path.exists(args.font_path):
        try:
            font_prop = FontProperties(fname=args.font_path)
        except Exception:
            font_prop = None

    # 收集所有样例的行（与 notebook 的 res 相似）
    rows: List[Dict[str, Any]] = []
    models_path = [
        #"/home/llama/test-rlif/checkpoints/TTRL/0826-152919/global_step_240/actor_hf_model",
        #"/home/llama/test-rlif/checkpoints/rent/global_step_150/actor_hf_model",
        #"/home/llama/test-rlif/checkpoints/Intuitor/global_step_116/actor_hf_model",
        #"/home/llama/test-rlif/checkpoints/EMPO/0824-052825/global_step_2499/actor_hf_model",
        "/home/llama/test-rlif/Qwen2.5-Math-1.5B"
    ]
    models_name = [
            #"ttrl", 
            #"rent", 
            #"intuitor", 
            #"empo", 
            "origin"
        ]
    for ids, model_path in enumerate(models_path):
        bundle = load_model(model_path, tp_size=args.tp_size, max_model_len=args.max_model_len, gpu_mem_utilization=args.gpu_mem_util)
        try:
            for idx, row in df_sample.iterrows():
                pid = int(row.get("id", -1))
                problem = row["problem"]
                sol = row["answer"]
                prompt = format_prompt(problem)
                gen = generate_with_scores(bundle, prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
                ex_ans = extract_answer(gen["response_text"]) if gen["response_text"] else ""
                is_ok, pred_val = evaluate_correctness(sol, ex_ans)
                rows.append({
                    "id": pid,
                    "model_name": models_name[ids],
                    "model_path": bundle.path,
                    "response_text": gen["response_text"],
                    "extracted_answer": ex_ans,
                    "predicted_value": pred_val,
                    "is_correct": is_ok,
                    "entropy_series": gen["entropy_series"],
                    "avg_logprob": gen["avg_logprob"],
                })
        finally:
            # 释放该模型的 vLLM/HF 资源
            try:
                bundle.vllm.close()
            except Exception:
                pass
            try:
                bundle.scorer.close()
            except Exception:
                pass
            del bundle
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # 保存结果
    res_df = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_dir, f"amc12_entropy_{models_name[0]}.csv")
    res_df.to_csv(out_csv, index=False, encoding="utf-8")


    print(f"已保存 CSV: {out_csv}")

    # 绘图（按模型聚合，分别绘制正确/错误）
    plot_entropy_curves_per_model(res_df, plots_dir, font_prop=font_prop)
    print(f"已保存图像到: {plots_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="AMC12 批处理生成与按模型绘图（正确/错误）")
    # parser.add_argument("--model-paths", type=str, required=True,
    #                     help="逗号分隔的模型路径列表，如: /path/A,/path/B")
    parser.add_argument("--data-path", type=str, default="/home/llama/test-rlif/datasets/amc12/data/train-00000-of-00001.parquet")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu-mem-util", type=float, default=0.7)
    parser.add_argument("--font-path", type=str, default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)