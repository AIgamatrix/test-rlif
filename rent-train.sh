source ./set_path.sh

cd  ./rent-rl


python -m verl.trainer.main_ppo exps="[grpo, entropy, format, sampleval, aime]" trainer.logger=['console']