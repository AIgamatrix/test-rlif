source ~/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate empo
echo "当前激活的环境: $(conda info )"
source ./set_path.sh

cd  ./EMPO/verl

bash examples/ttrl/empo-math.sh