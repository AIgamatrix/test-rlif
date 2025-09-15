source ~/miniconda3/etc/profile.d/conda.sh

conda deactivate

conda activate empo
echo "当前激活的环境: $(conda info )"

source ./set_path.sh
cd  ./rent-rl
python -m verl.trainer.main_ppo exps="[grpo, entropy, format, sampleval, aime]" trainer.logger=['console']