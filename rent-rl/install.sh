conda create -n verl python=3.12
conda activate verl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.3
pip install flash-attn --no-build-isolation
pip install -e ".[vllm]"
conda install -c conda-forge libstdcxx-ng
pip install ipdb
