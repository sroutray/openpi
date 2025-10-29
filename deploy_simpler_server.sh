#!/bin/bash

export NCCL_P2P_DISABLE=1

source /usr/share/Modules/init/bash

#!/bin/bash
cuda_version=12.4

# Load the CUDA module
module load "cuda-${cuda_version}"

# Set the environment variables
export CUDA_HOME="/usr/local/cuda-${cuda_version}"

# Load MPI
module load mpi/openmpi-x86_64

# Huggingface
export HF_TOKEN=""

export HF_HOME="/data/hf_cache"
export HF_DATASETS_CACHE="/data/hf_cache"
export TRANSFORMERS_CACHE="/data/hf_cache"

# openpi uv setup
export PATH=/home/jasonl6/miniconda3/envs/pi0/bin:$PATH
source .venv/bin/activate

export OPENPI_DATA_HOME="/data/user_data/jasonl6/sandeep/Work/openpi"
export LEROBOT_HOME="/home/jasonl6/sandeep/Work/lq-exp/openpi/data"

uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_processed_v5_low_mem_finetune --policy.dir=/home/jasonl6/sandeep/Work/lq-exp/openpi/checkpoints/pi0_processed_v5_low_mem_finetune/processed_v5-ft-run1/45000
