#!/usr/bin/env python3
"""
将 FSDP Actor 检查点合并为 Hugging Face 模型目录。

用法示例：
python entropy_metrics/convert_fsdp_actor_to_hf.py \
  --actor-ckpt-dir /home/llama/test-rlif/checkpoints/MYRLIF/1105-165007/global_step_240/actor \
  --base-hf-model /home/llama/test-rlif/Qwen2.5-Math-1.5B \
  --target-dir /home/llama/test-rlif/test_res_qwen/MYRLIF/merged_hf_actor
"""

import os
import sys
import time
import argparse
import subprocess


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor-ckpt-dir", type=str, required=True, help="FSDP Actor 检查点目录")
    p.add_argument("--base-hf-model", type=str, required=True, help="基础 HF 模型目录 (用于读取 config)")
    p.add_argument("--target-dir", type=str, required=True, help="输出 HF 模型目录")
    p.add_argument("--model-merger-path", type=str, default="", help="可选：覆盖默认合并脚本路径")
    args = p.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_merger = os.path.join(repo_root, "MyRLIF", "verl", "scripts", "model_merger.py")
    merger_py = args.model_merger_path or default_merger
    if not os.path.exists(merger_py):
        raise FileNotFoundError(f"未找到合并脚本: {merger_py}")

    ensure_dir(args.target_dir)
    has_config = os.path.exists(os.path.join(args.target_dir, "config.json"))
    has_weights = os.path.exists(os.path.join(args.target_dir, "pytorch_model.bin")) or os.path.exists(
        os.path.join(args.target_dir, "model.safetensors")
    )
    if has_config and has_weights:
        print(f"目标目录已存在合并结果，跳过：{args.target_dir}")
        return

    cmd = [
        sys.executable,
        merger_py,
        "--backend",
        "fsdp",
        "--hf_model_path",
        args.base_hf_model,
        "--local_dir",
        args.actor_ckpt_dir,
        "--target_dir",
        args.target_dir,
    ]
    print(f"[merge] Running: {' '.join(cmd)}")
    start = time.time()
    subprocess.run(cmd, check=True)
    print(f"[merge] Done in {time.time() - start:.1f}s. Saved to {args.target_dir}")


if __name__ == "__main__":
    main()