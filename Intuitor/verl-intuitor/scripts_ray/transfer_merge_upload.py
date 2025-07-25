#!/usr/bin/env python3

import os
import subprocess
import shutil
from huggingface_hub import HfApi, upload_folder

ray_tmp_folder = YOUR_RAY_TMP_FOLDER

# Hardcoded configuration
# input_ckpt_dir = "f{ray_tmp_folder}/session_2025-06-26_00-31-08_398079_2873595/runtime_resources/working_dir_files/_ray_pkg_e7b17b5dc6b9c6b6/checkpoints/verl_aosong/math_intuitor_qwen25-7b/global_step_58/actor"
# output_dir = "/path/to/Intuitor/ckpts/math_intuitor_qwen25-7b"
# nodes = [NODE11,NODE12]
# hf_repo_id = "your_hf_username/math_intuitor_qwen25-7b"


# input_ckpt_dir = "f{ray_tmp_folder}/session_latest/runtime_resources/working_dir_files/_ray_pkg_932097416eb4556d/checkpoints/verl_aosong/math_grpo_qwen25-14b/global_step_58/actor"
# output_dir = "/path/to/Intuitor/ckpts/math_grpo_qwen25-14b"
# nodes = [NODE5,NODE6,NODE7,NODE8]
# hf_repo_id = "your_hf_username/math_grpo_qwen25-14b"


input_ckpt_dir = "f{ray_tmp_folder}/session_latest/runtime_resources/working_dir_files/_ray_pkg_36267e5a4b300547/checkpoints/verl_aosong/math_intuitor_qwen25-14b/global_step_58/actor/"
output_dir = "/path/to/Intuitor/ckpts/math_intuitor_qwen25-14b"
nodes = [NODE1, NODE2, NODE3, NODE4]
hf_repo_id = "your_hf_username/math_intuitor_qwen25-14b"


# input_ckpt_dir = "f{ray_tmp_folder}/session_latest/runtime_resources/working_dir_files/_ray_pkg_380026430aef1571/checkpoints/verl_aosong/math_grpo_qwen25-7b/global_step_58/actor"
# output_dir = "/path/to/Intuitor/ckpts/math_grpo_qwen25-7b"
# nodes = [NDOE9, NODE10]
# hf_repo_id = "your_hf_username/math_grpo_qwen25-7b"


# ============================================================================================
# HuggingFace configuration
hf_token = "hf_...."  # Replace with your actual token
hf_private = True
hf_commit_message = "Upload merged checkpoint"


action_transfer_files = True  # Set to False to skip file transfer
action_delete_sharded = False  # Set to False to skip deletion of intermediate files
action_merge_files = True  # Set to False to skip merging files
action_upload_files = True  # Set to False to skip uploading to HuggingFace



all_sharded_dir = os.path.join(output_dir, "all_sharded_ckpts")

if action_transfer_files:
    # Create directories
    os.makedirs(all_sharded_dir, exist_ok=True)

    # Rsync from each node
    for node in nodes:
        print(f"\nCopying from {node}...")
        rsync_cmd = [
            "rsync", "-avz", "--progress",
            "-e", "ssh -o StrictHostKeyChecking=no",
            "--exclude", "*extra_state_*",
            "--exclude", "*optim_*",
            f"{node}:{input_ckpt_dir}/",
            f"{all_sharded_dir}/"
        ]

        subprocess.run(rsync_cmd)

if action_merge_files:
    # Run model merger
    print("\nMerging checkpoint...")
    merge_cmd = [
        "/path/to/Intuitor/verl-intuitor/env_verl/bin/python",
        "/path/to/Intuitor/verl-intuitor/scripts/model_merger.py",
        "merge",
        "--backend", "fsdp",
        "--local_dir", all_sharded_dir,
        "--target_dir", output_dir
    ]

    subprocess.run(merge_cmd)

if action_delete_sharded:
    # Delete intermediate directory
    print("\nCleaning up...")
    shutil.rmtree(all_sharded_dir)

    print(f"\nDone! Merged checkpoint saved to: {output_dir}")



if action_upload_files:
    # Upload to HuggingFace
    print(f"\nUploading to HuggingFace: {hf_repo_id}")

    # Initialize HF API
    api = HfApi(token=hf_token)

    api.create_repo(
        repo_id=hf_repo_id,
        private=hf_private,
        exist_ok=True
    )
    print(f"Repository ready: {hf_repo_id}")

    upload_folder(
        folder_path=output_dir,
        repo_id=hf_repo_id,
        commit_message=hf_commit_message,
        token=hf_token,
        ignore_patterns=["all_sharded_ckpts/**"]
    )
    print(f"\n✅ Successfully uploaded to https://huggingface.co/{hf_repo_id}")

    print("\nAll done!")