#!/bin/bash

# Sync code to nh2
unset LD_LIBRARY_PATH

# rsync -avz --exclude '.git' \
#     --exclude '*.pyc' \
#     --exclude '__pycache__' \
#     --exclude '.pytest_cache' \
#     --exclude 'logs' \
#     --exclude 'wandb' \
#     --exclude 'outputs' \
#     --exclude 'data' \
#     ./ nh2:~/phd_projects/verl/


rsync -avz --exclude '.git' \
    --exclude '*.pyc' \
    --exclude '__pycache__' \
    --exclude '.pytest_cache' \
    --exclude 'logs' \
    --exclude 'checkpoints' \
    --exclude 'wandb' \
    --exclude 'outputs' \
    --exclude 'data' \
    ./ slip:~/phd_projects/verl/