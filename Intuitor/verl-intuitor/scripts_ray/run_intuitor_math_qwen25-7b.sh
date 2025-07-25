
set +x
source ./ray_utils.sh
HEAD_NODE=NODE11
TOTAL_NODES=2
# Get dashboard address
DASHBOARD_ADDRESS="http://${HEAD_NODE}:${RAY_DASHBOARD_PORT}"
echo "Ray dashboard address: $DASHBOARD_ADDRESS"

# Submit the job
echo "Submitting Ray job to $DASHBOARD_ADDRESS"
EXPERIMENT_NAME="math_intuitor_qwen25-7b"

/path/to/Intuitor/verl-intuitor/env_verl/bin/ray job submit \
    --address="$DASHBOARD_ADDRESS" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    /path/to/Intuitor/verl-intuitor/env_verl/bin/python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=intuitor \
    data.train_files=$PWD/data/math/train.parquet \
    data.val_files=$PWD/data/math/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=512 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.005 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$TOTAL_NODES \
    trainer.logger=['console','wandb'] \
    trainer.project_name=verl_exp \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=1 2>&1 | tee log_${EXPERIMENT_NAME}.log