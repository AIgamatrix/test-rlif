python -m verl.trainer.main_ppo exps="[grpo, entropy,gsm8k]"

# to_run:
python -m verl.trainer.main_ppo exps="[grpo, entropy,gsm8k]" base_model=Qwen/Qwen2.5-0.5B-Instruct
python -m verl.trainer.main_ppo exps="[grpo, entropy,gsm8k]" base_model=Qwen/Qwen2.5-1.5B-Instruct
python -m verl.trainer.main_ppo exps="[grpo, entropy,gsm8k]" base_model=Qwen/Qwen2.5-7B-Instruct




python -m verl.trainer.main_ppo exps="[grpo, sampleval, smallmbs, gsm8k, entropy, format]" 
# grpo, 