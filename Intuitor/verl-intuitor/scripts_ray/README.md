# Verl Multi-node Training with Ray (based on Verl 0.3.1.dev)

**Author:** Aosong Feng (aosong.feng@yale.edu)

This guide expands on the official Verl [documentation](https://verl.readthedocs.io/en/latest/start/multinode.html) for multi-node training. It provides additional details for launching multi-node training using the Intuitor Verl version.

---

## Cluster Setup

We provide several utility functions in `ray_utils.sh` which can be sourced and reused in other bash scripts. Update the default configurations (lines 7–9) with your specific machine setup.

To launch a cluster with 4 nodes (on SLURM), run:

```bash
bash setup_ray_cluster_1.sh
```

Modify the following variables as needed:

1. **`HEAD_NODE`, `WORKER_NODE`** – IP or hostname of the head and worker nodes.  
2. **`ACTIVATION_PATH`** – Path to your environment activation script (e.g., `/path/to/your_venv/bin/activate` or `/path/to/your_conda/bin/activate`).  
3. **`RAY_SSH_USER`** – Username for SSH access across all nodes (must be the same for all).  
4. **`RAY_TMPDIR`** – Ray stores temporary data in `/tmp/ray` by default. If your root partition is small, set this to a directory on a larger disk.

To run multiple experiments in parallel, continue with:

- `setup_ray_cluster_2.sh` (4 nodes)  
- `setup_ray_cluster_3.sh` (2 nodes)  
- `setup_ray_cluster_4.sh` (2 nodes)

---

## Launch Training

To train GRPO on the math dataset using Qwen2.5‑14B, run:

```bash
bash run_grpo_math_qwen25-14b.sh
```

Update the script as follows:

1. Replace `/path/to` with the correct environment paths to ensure the intended `ray` and `python` executables are used.  
2. Set `HEAD_NODE` and `TOTAL_NODES` to target the intended cluster.  
3. Environment variables such as `wandb_key` must be set in `verl/trainer/runtime_env.yaml`; setting them in the bash script will not propagate correctly.  
4. Monitor training via the Ray dashboard (see the official tutorial) or check logs in your local `RAY_TMPDIR` directory. If using SSH, configure port forwarding to view the dashboard locally (VS Code sometimes does this automatically).

---

## Checkpoint Aggregation

The scripts assume each Ray cluster has 4 nodes, each with 8 × 40 GB GPUs. With FSDP, the 14B model is sharded into 32 pieces:

- **NODE1** hosts shards 1–8  
- **NODE2** hosts shards 9–16  
- … and so on.

Before aggregation, transfer all shard files to a single machine. Then run the checkpoint merge script to convert them into a HuggingFace‑compatible `safetensors` model.

An example script, `transfer_merge_upload.py`, performs the following:

1. Transfers all shards to one machine.  
2. Runs `scripts/model_merger.py`.  
3. Deletes the transferred shards.  
4. Uploads the merged model to a private HuggingFace repository.

---
