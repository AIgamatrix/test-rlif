#!/usr/bin/env python3
"""
使用已合并的 HF Actor 模型评估 AMC12 数据集（简洁版）。

用法示例：
python entropy_metrics/test_amc12_with_hf_actor.py \
  --hf-actor /home/llama/test-rlif/test_res_qwen/MYRLIF/merged_hf_actor \
  --base-hf-tokenizer /home/llama/test-rlif/Qwen2.5-Math-1.5B \
  --data-path /home/llama/test-rlif/datasets/amc12/data/train-00000-of-00001.parquet \
  --output-dir /home/llama/test-rlif/test_res_qwen/MYRLIF \
  --max-new-tokens 512 --temperature 0.6
"""

import os
import re
import json
import argparse
from typing import Any, Dict, List, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def format_prompt(tokenizer: AutoTokenizer, problem: str) -> str:
    """格式化问题提示（简洁实现）"""
    prompt = (
        "请解决以下数学问题，并给出最终答案。请仔细思考，然后再末尾提供最终答案。\n"
        "请将最终答案包裹在\\boxed{}中。\n"
        f"问题：{problem}。\n"
    )
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return text
        except Exception:
            return prompt
    return prompt


def extract_answer(response: str) -> str:
    """从模型响应中提取答案（按你给的写法）"""
    match = re.search(r"\\boxed\{(.+?)\}", response)
    if match:
        group_count = len(match.groups())
        last_group = match.group(group_count).strip()
        return last_group
    return ""


def load_amc12_df(data_path: str) -> pd.DataFrame:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"AMC12数据集未找到: {data_path}")
    df = pd.read_parquet(data_path)
    if not set(["id", "problem", "answer"]).issubset(df.columns):
        raise ValueError("AMC12数据集缺少必要列: id/problem/answer")
    return df


def run_single_test(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str, problem: str, correct_answer: float, max_new_tokens: int, temperature: float) -> Dict[str, Any]:
    """运行单个测试（简洁实现）"""
    prompt_text = format_prompt(tokenizer, problem)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=max(0.0, temperature),
            do_sample=temperature > 0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_ids, skip_special_tokens=True)
    extracted_answer = extract_answer(response)
    is_correct = False
    predicted_value = None
    try:
        if extracted_answer != "":
            predicted_value = float(extracted_answer)
            is_correct = (float(correct_answer) == predicted_value)
    except Exception:
        pass
    return {
        "problem": problem,
        "correct_answer": float(correct_answer),
        "model_response": response,
        "extracted_answer": extracted_answer,
        "predicted_value": predicted_value,
        "is_correct": bool(is_correct),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-actor", type=str, required=True, help="已合并的 HF Actor 模型目录")
    p.add_argument("--base-hf-tokenizer", type=str, required=True, help="基础 HF 模型目录（用于 tokenizer）")
    p.add_argument("--data-path", type=str, required=True, help="AMC12 数据 parquet 路径")
    p.add_argument("--output-dir", type=str, required=True, help="输出目录")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.6)
    args = p.parse_args()

    ensure_dir(args.output_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.base_hf_tokenizer, trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_actor,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model.to(device)
    model.eval()

    df = load_amc12_df(args.data_path)

    detailed: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        problem = str(row["problem"])  # 原数据可能包含 LaTeX
        correct = float(row["answer"])  # AMC12 为数值答案
        try:
            item = run_single_test(
                model,
                tokenizer,
                device,
                problem,
                correct,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        except Exception as e:
            item = {
                "problem": problem,
                "correct_answer": correct,
                "model_response": f"[生成失败] {e}",
                "extracted_answer": "",
                "predicted_value": None,
                "is_correct": False,
            }
        detailed.append(item)

    total = len(detailed)
    correct_count = sum(1 for x in detailed if x["is_correct"]) if total > 0 else 0
    accuracy = (correct_count / total) if total > 0 else 0.0

    results_path = os.path.join(args.output_dir, "myrlif_amc12_test_results.json")
    summary_path = os.path.join(args.output_dir, "myrlif_amc12_test_results_summary.json")

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2)

    summary_obj = {
        "model_path": args.hf_actor,
        "total_questions": total,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "detailed_results": detailed,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    print(f"写入完成：\n- {results_path}\n- {summary_path}")


if __name__ == "__main__":
    main()