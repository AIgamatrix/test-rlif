#!/usr/bin/env python3
"""
批量将 FSDP checkpoints (TTRL / Intuitor / EMPO) 转换为 HuggingFace 格式，并进行验证与四套基准测试（AIME/AMC12/GPQA/Minerva）。

使用说明：
  python run_convert_and_tests.py \
    --convert        # 先进行FSDP->HF转换
    --verify         # 转换后进行模型加载与简单推理验证（通过 verify_hf_models.py）
    --eval           # 对三模型分别在四套数据集上跑测试（通过四个测试脚本），生成12份结果

可选参数（如不提供，将自动探测并给出合理默认值）：
  --ckpt-root /home/llama/test-rlif/checkpoints
  --out-root  /home/llama/test-rlif/test_res_qwen
  --aime-path /home/llama/test-rlif/datasets/aime_val/data/train-00000-of-00001.parquet
  --amc-path  /home/llama/test-rlif/datasets/amc12/data/train-00000-of-00001.parquet
  --gpqa-path /home/llama/test-rlif/datasets/gpqa/gpqa_main.csv
  --minerva-path /home/llama/test-rlif/datasets/minerva/test.jsonl

注意：
- 将尽量重用各 checkpoint 步内的 actor_hf 作为 --hf_model_path（若没有，则回退使用 EMPO 的 actor_hf 配置）。
- 验证与评测阶段均通过独立脚本以子进程方式调用，避免重复实现。
- 若某数据集路径不存在，将跳过相应评测并给出告警。
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import shutil

# ---------------------------
# 工具函数：路径探测与选取
# ---------------------------

def _read_latest_step(txt_path: Path) -> Optional[int]:
    try:
        if txt_path.exists():
            txt = txt_path.read_text().strip()
            m = re.search(r"(\d+)", txt)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def _find_latest_global_step_dir(base_dir: Path) -> Optional[Path]:
    """返回 base_dir 下最新的 global_step_xxx 目录。"""
    if not base_dir.exists():
        return None
    step_dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("global_step_")]
    if not step_dirs:
        return None
    def step_num(p: Path) -> int:
        m = re.search(r"(\d+)$", p.name)
        return int(m.group(1)) if m else -1
    step_dirs.sort(key=step_num)
    return step_dirs[-1]


def _find_latest_subfolder_with_txt(root: Path) -> Optional[Path]:
    """对于类似 EMPO/0824-052825/ 结构，优先使用 latest_checkpointed_iteration.txt 指示的步。"""
    if not root.exists():
        return None
    # 如果 root 下只有一步层级（如 Intuitor/ 直接有 global_step_XX），直接在 root 里找
    direct_steps = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("global_step_")]
    if direct_steps:
        latest_step = _read_latest_step(root / "latest_checkpointed_iteration.txt")
        if latest_step is not None:
            cand = root / f"global_step_{latest_step}"
            return cand if cand.exists() else _find_latest_global_step_dir(root)
        return _find_latest_global_step_dir(root)

    # 否则进入子文件夹（如 0824-052825）
    subfolders = [p for p in root.iterdir() if p.is_dir()]
    if not subfolders:
        return None
    # 选择最近修改的子目录
    subfolders.sort(key=lambda p: p.stat().st_mtime)
    sub = subfolders[-1]
    latest_step = _read_latest_step(sub / "latest_checkpointed_iteration.txt")
    if latest_step is not None:
        cand = sub / f"global_step_{latest_step}"
        if cand.exists():
            return cand
    return _find_latest_global_step_dir(sub)


def _pick_hf_config_dir(preferred: Optional[Path], empo_default: Optional[Path], fallback_target: Optional[Path] = None) -> Path:
    """优先使用当前步的 actor_hf，其次回退到 EMPO 的 actor_hf；
    若均不存在，则在已转换过的 target 目录中查找（需要包含config.json）。
    """
    candidates = []
    if preferred:
        candidates.append(preferred)
    if empo_default and empo_default not in candidates:
        candidates.append(empo_default)
    if fallback_target and fallback_target not in candidates:
        candidates.append(fallback_target)
    for c in candidates:
        if c and c.exists() and (c / "config.json").exists():
            return c
    raise FileNotFoundError("无法找到可用的HF配置目录（需要包含config.json）。请检查actor_hf是否存在，或手动指定--hf-model-path。")


@dataclass
class ModelEntry:
    name: str
    step_dir: Path          # .../global_step_xxx
    actor_dir: Path         # .../global_step_xxx/actor
    hf_config_dir: Path     # 用作 --hf_model_path
    target_hf_dir: Path     # 转换输出目录


# ---------------------------
# 转换：调用 TTRL/verl/scripts/model_merger.py
# ---------------------------

def _ensure_tokenizer_files(src_dir: Path, dst_dir: Path) -> None:
    """确保dst_dir内包含必要的tokenizer相关文件，从src_dir复制缺失项。
    不会覆盖已存在文件。
    """
    if not src_dir or not src_dir.exists():
        print(f"[tokenizer] 源目录不存在，跳过: {src_dir}")
        return
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
    # 可能用到的文件名
    candidates = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "ice_text.model",
        "generation_config.json",
    ]
    for name in candidates:
        s = src_dir / name
        d = dst_dir / name
        if s.exists() and not d.exists():
            try:
                shutil.copy2(s, d)
                print(f"[tokenizer] 已复制 {name} -> {d}")
            except Exception as e:
                print(f"[tokenizer] 复制失败 {s} -> {d}: {e}")


def convert_fsdp_checkpoint(entry: ModelEntry, merger_py: Path) -> None:
    cmd = [
        sys.executable, str(merger_py),
        "--backend", "fsdp",
        "--hf_model_path", str(entry.hf_config_dir),
        "--local_dir", str(entry.actor_dir),
        "--target_dir", str(entry.target_hf_dir),
    ]
    print(f"\n[转换] {entry.name}:\n  local_dir   = {entry.actor_dir}\n  hf_config   = {entry.hf_config_dir}\n  target_dir  = {entry.target_hf_dir}\n  cmd         = {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # 确保目标目录内具备tokenizer文件，便于下游脚本直接加载
    _ensure_tokenizer_files(entry.hf_config_dir, entry.target_hf_dir)


# ---------------------------
# 验证：加载与简短推理（改为调用独立脚本 verify_hf_models.py）
# ---------------------------

def verify_models_by_script(entries: List[ModelEntry], verify_script: Path) -> None:
    if not verify_script.exists():
        print(f"✗ 未找到验证脚本: {verify_script}")
        return
    # 进入验证前再次确保tokenizer存在
    for e in entries:
        _ensure_tokenizer_files(e.hf_config_dir, e.target_hf_dir)
    # 组装 --models 参数：name:path
    models_arg: List[str] = [f"{e.name}:{e.target_hf_dir}" for e in entries]
    cmd = [sys.executable, str(verify_script), "--models", *models_arg]
    print(f"\n[验证] 调用: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"验证脚本返回非零退出码: {e.returncode}")


# ---------------------------
# 评测：通过四个外部测试脚本
# ---------------------------

def run_test_script(script_path: Path, model_path: Path, data_path: Path, output_path: Path) -> bool:
    if not script_path.exists():
        print(f"✗ 未找到测试脚本: {script_path}")
        return False
    if not data_path.exists():
        print(f"[跳过] 数据文件不存在: {data_path}")
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(script_path),
        "--model-path", str(model_path),
        "--data-path", str(data_path),
        "--output-path", str(output_path),
    ]
    print(f"调用测试脚本: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"测试脚本失败({script_path.name})，退出码: {e.returncode}")
        return False


# ---------------------------
# 主流程
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="/home/llama/test-rlif/checkpoints", type=str)
    ap.add_argument("--out-root", default="/home/llama/test-rlif/test_res_qwen", type=str)
    ap.add_argument("--convert", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--hf-model-path", default=None, type=str, help="手动指定HF配置目录（包含config.json），将覆盖自动探测")
    # 数据路径
    ap.add_argument("--aime-path", default="/home/llama/test-rlif/datasets/aime_val/data/train-00000-of-00001.parquet")
    ap.add_argument("--amc-path", default="/home/llama/test-rlif/datasets/amc12/data/train-00000-of-00001.parquet")
    ap.add_argument("--gpqa-path", default="/home/llama/test-rlif/datasets/gpqa/gpqa_main.csv")
    ap.add_argument("--minerva-path", default="/home/llama/test-rlif/datasets/minerva/test.jsonl")
    args = ap.parse_args()

    ckpt_root = Path(args.ckpt_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # 找到三种方法的最新checkpoint步
    empo_step = _find_latest_subfolder_with_txt(ckpt_root / "EMPO")
    ttrl_step = _find_latest_subfolder_with_txt(ckpt_root / "TTRL")
    intuit_step = _find_latest_subfolder_with_txt(ckpt_root / "Intuitor")
    rent_step = _find_latest_subfolder_with_txt(ckpt_root / "rent")

    if not empo_step:
        print("✗ 未找到 EMPO 的 global_step 目录")
    if not ttrl_step:
        print("✗ 未找到 TTRL 的 global_step 目录")
    if not intuit_step:
        print("✗ 未找到 Intuitor 的 global_step 目录")

    # EMPO 的 actor_hf（若存在）用作默认的 HF 配置兜底
    empo_actor_hf = empo_step / "actor_hf" if empo_step else None

    def build_entry(name: str, step: Optional[Path]) -> Optional[ModelEntry]:
        if not step:
            return None
        actor = step / "actor"
        if not actor.exists():
            print(f"✗ {name} 缺少 actor 目录: {actor}")
            return None
        preferred_hf_cfg = (step / "actor_hf") if (step / "actor_hf").exists() else None
        target = step / "actor_hf_model"
        target.mkdir(parents=True, exist_ok=True)
        hf_cfg = Path(args.hf_model_path) if args.hf_model_path else _pick_hf_config_dir(preferred_hf_cfg, empo_actor_hf, target)
        return ModelEntry(name=name, step_dir=step, actor_dir=actor, hf_config_dir=hf_cfg, target_hf_dir=target)

    entries: List[ModelEntry] = []
    #for nm, st in [("EMPO", empo_step), ("TTRL", ttrl_step), ("Intuitor", intuit_step)]:
    for nm, st in [("Rent", rent_step)]:
        ent = build_entry(nm, st)
        if ent:
            entries.append(ent)
    if not entries:
        print("没有可用的模型条目，退出。")
        return

    # 路径到转换脚本
    merger_py = Path("/home/llama/test-rlif/TTRL/verl/scripts/model_merger.py")
    if not merger_py.exists():
        print(f"✗ 未找到转换脚本: {merger_py}")
        return

    # 1) 转换
    if args.convert:
        for e in entries:
            convert_fsdp_checkpoint(e, merger_py)

    # 2) 验证（外部脚本）
    if args.verify:
        verify_script = Path("/home/llama/test-rlif/verify_hf_models.py")
        verify_models_by_script(entries, verify_script)

    # 3) 评测（外部脚本）
    if args.eval:
        # 四个测试脚本的路径
        aime_py = Path("/home/llama/test-rlif/test_qwen_aime.py")
        amc_py = Path("/home/llama/test-rlif/test_qwen_amc.py")
        gpqa_py = Path("/home/llama/test-rlif/test_qwen_gpqa.py")
        minerva_py = Path("/home/llama/test-rlif/test_qwen_minerva.py")

        for e in entries:
            print("\n" + "=" * 80)
            print(f"开始评测: 模型={e.name}({e.target_hf_dir})")
            out_dir = out_root / e.name
            out_dir.mkdir(parents=True, exist_ok=True)
            # 确保tokenizer文件存在
            _ensure_tokenizer_files(e.hf_config_dir, e.target_hf_dir)

            # # AIME
            # run_test_script(
            #     aime_py,
            #     e.target_hf_dir,
            #     Path(args.aime_path),
            #     out_dir / "aime_results.json",
            # )
            # # AMC12
            # run_test_script(
            #     amc_py,
            #     e.target_hf_dir,
            #     Path(args.amc_path),
            #     out_dir / "amc12_results.json",
            # )
            # GPQA
            run_test_script(
                gpqa_py,
                e.target_hf_dir,
                Path(args.gpqa_path),
                out_dir / "gpqa_results.json",
            )
            # Minerva
            run_test_script(
                minerva_py,
                e.target_hf_dir,
                Path(args.minerva_path),
                out_dir / "minerva_results.json",
            )


if __name__ == "__main__":
    # 延迟导入：避免未使用时报错
    import re  # noqa: F401  # 被_eval函数使用
    main()