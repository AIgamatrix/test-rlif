source ~/miniconda3/etc/profile.d/conda.sh

conda deactivate

conda activate empo
echo "当前激活的环境: $(conda info )"
source ./set_path.sh

cd  ./MyRLIF/verl

bash examples/ttrl/Qwen-2.5-Math/aime.sh
