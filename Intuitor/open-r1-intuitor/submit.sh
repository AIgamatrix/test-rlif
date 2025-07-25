
sbatch --nodes=5 ./slurm/run_7b_grpo.slurm Qwen2.5-7B grpo demo zero3

sbatch --nodes=5 ./slurm/run_7b_spo.slurm Qwen2.5-7B intuitor demo zero3

sbatch --nodes=9 ./slurm/run_14b_grpo.slurm Qwen2.5-14B grpo demo zero3

sbatch --nodes=9 ./slurm/run_14b_spo.slurm Qwen2.5-14B intuitor demo zero3