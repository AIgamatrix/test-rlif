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


def load_model(model_path: str, tp_size: int = 1, max_model_len: int = 4096, gpu_mem_utilization: float = 0.7, enforce_eager: bool = True) -> ModelBundle:
    from transformers import AutoTokenizer, AutoConfig
    # tokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # vLLM 生成器
    vllm = VLLMGenerator(
        model_path=model_path,
        tensor_parallel_size=tp_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem_utilization,
        enforce_eager=enforce_eager,
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
            out_png = os.path.join(save_dir, f"{model_name}_correct_entropy4.png")
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
            out_png = os.path.join(save_dir, f"{model_name}_incorrect_entropy4.png")
            plt.savefig(out_png, dpi=300, bbox_inches="tight")
            plt.close()


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                pass
    return data


def run_worker(args, model_path: str) -> str:
    df = load_amc12_df(args.data_path)
    df_sample = df.iloc[:args.num_samples, :].copy() if args.num_samples > 0 else df
    name = os.path.basename(model_path.rstrip("/"))
    worker_out_dir = os.path.join(args.output_dir, "workers", name)
    os.makedirs(worker_out_dir, exist_ok=True)
    out_jsonl = os.path.join(worker_out_dir, f"{name}.jsonl")

    bundle = load_model(model_path, tp_size=args.tp_size, max_model_len=args.max_model_len,
                        gpu_mem_utilization=args.gpu_mem_util, enforce_eager=args.enforce_eager)
    rows: List[Dict[str, Any]] = []
    try:
        for _, row in df_sample.iterrows():
            pid = int(row.get("id", -1))
            problem = row["problem"]
            sol = row["answer"]
            prompt = format_prompt(problem)
            gen = generate_with_scores(bundle, prompt, args.max_new_tokens, args.temperature, args.top_p, args.top_k)
            ex_ans = extract_answer(gen["response_text"]) if gen["response_text"] else ""
            is_ok, pred_val = evaluate_correctness(sol, ex_ans)
            rows.append({
                "id": pid,
                "model_name": name,
                "model_path": bundle.path,
                "response_text": gen["response_text"],
                "extracted_answer": ex_ans,
                "predicted_value": pred_val,
                "is_correct": is_ok,
                "entropy_series": gen["entropy_series"],
                "avg_logprob": gen["avg_logprob"],
            })
    finally:
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

    write_jsonl(out_jsonl, rows)
    return out_jsonl


def run(args):
    # 可选：设置 CUDA 显存分配策略，减少碎片化
    if args.cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    model_paths = [p.strip() for p in args.model_paths.split(",") if p.strip()] if args.model_paths else []
    os.makedirs(args.output_dir, exist_ok=True)

    # 字体属性
    font_prop = None
    if args.font_path and os.path.exists(args.font_path):
        try:
            font_prop = FontProperties(fname=args.font_path)
        except Exception:
            font_prop = None

    # 如果是子进程（worker）模式，仅处理一个模型并退出
    if args.worker_model_path:
        out_jsonl = run_worker(args, args.worker_model_path)
        print(f"[worker] 已保存 JSONL: {out_jsonl}")
        return

    # 协调器模式：逐模型启动子进程，完成后汇总与绘图
    import subprocess, sys
    all_rows: List[Dict[str, Any]] = []
    workers_dir = os.path.join(args.output_dir, "workers")
    os.makedirs(workers_dir, exist_ok=True)

    this_script = os.path.abspath(__file__)
    base_cmd = [sys.executable, this_script,
                "--data-path", args.data_path,
                "--num-samples", str(args.num_samples),
                "--max-new-tokens", str(args.max_new_tokens),
                "--temperature", str(args.temperature),
                "--top_p", str(args.top_p),
                "--top_k", str(args.top_k),
                "--tp_size", str(args.tp_size),
                "--max_model_len", str(args.max_model_len),
                "--gpu-mem-util", str(args.gpu_mem_util),
                "--font-path", args.font_path,
                "--output-dir", args.output_dir,
                "--cuda-alloc-conf", args.cuda_alloc_conf or "",
                ]
    if args.enforce_eager:
        base_cmd.append("--enforce-eager")

    for mp in model_paths:
        cmd = base_cmd + ["--worker-model-path", mp]
        env = os.environ.copy()
        if args.cuda_alloc_conf:
            env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf
        print(f"[coordinator] 启动子进程处理模型: {mp}")
        ret = subprocess.run(cmd, env=env)
        if ret.returncode != 0:
            print(f"[coordinator] 子进程失败: {mp}, returncode={ret.returncode}")
            continue
        # 读取该模型生成的 JSONL
        name = os.path.basename(mp.rstrip("/"))
        out_jsonl = os.path.join(workers_dir, name, f"{name}.jsonl")
        rows = read_jsonl(out_jsonl)
        all_rows.extend(rows)

    # 汇总保存
    res_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(args.output_dir, "amc12_entropy_results_flat.csv")
    res_df.to_csv(out_csv, index=False, encoding="utf-8")
    out_json = os.path.join(args.output_dir, "amc12_entropy_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "data_path": args.data_path,
            "num_samples": args.num_samples,
            "results": all_rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"[coordinator] 已保存 CSV: {out_csv}")
    print(f"[coordinator] 已保存 JSON: {out_json}")

    # 按模型绘图
    plots_dir = os.path.join(args.output_dir, "plots_per_model")
    os.makedirs(plots_dir, exist_ok=True)
    plot_entropy_curves_per_model(res_df, plots_dir, font_prop=font_prop)
    print(f"[coordinator] 已保存图像到: {plots_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="AMC12 批处理生成与按模型绘图（正确/错误）")
    parser.add_argument("--model-paths", type=str, required=True,
                        help="逗号分隔的模型路径列表，如: /path/A,/path/B")
    parser.add_argument("--data-path", type=str, default="/home/llama/test-rlif/datasets/amc12/data/train-00000-of-00001.parquet")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu-mem-util", type=float, default=0.7)
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="启用 eager，避免 torch.compile 的图优化占用显存")
    parser.add_argument("--cuda-alloc-conf", type=str, default="expandable_segments:True",
                        help="设置 PYTORCH_CUDA_ALLOC_CONF 以减少碎片化。留空则不设置")
    parser.add_argument("--font-path", type=str, default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    parser.add_argument("--output-dir", type=str, default="./entropy_metrics/outputs_batch")
    # worker 专用参数：用于子进程单模型处理
    parser.add_argument("--worker-model-path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
