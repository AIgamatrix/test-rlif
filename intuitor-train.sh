source ~/miniconda3/etc/profile.d/conda.sh

conda deactivate 

conda activate intuit
echo "当前激活的环境: $(conda info )"
source ./set_path.sh

cd  ./Intuitor/verl-intuitor

bash math_intuitor.sh
