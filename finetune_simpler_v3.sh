#!/bin/bash
#SBATCH --job-name=op_simpl
#SBATCH --output=jobs/op_simpl.%j.out
#SBATCH --error=jobs/op_simpl.%j.err
#SBATCH --partition=preempt
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=3
#SBATCH --mem=220G
#SBATCH --time=1-23:59:00
#SBATCH --constraint='L40S|H100|A100_80GB|A6000|6000Ada'
#SBATCH --exclude='shire-1-1'
##exclude='babel-12-21,babel-13-1'
###exclude='babel-14-25,babel-11-5,babel-1-23,babel-13-13,babel-4-13'
##constraint='L40S|H100|A100_80GB|A6000|6000Ada'

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

export WANDB_USER_NAME="sroutray"
export WANDB_API_KEY=""
wandb login $WANDB_API_KEY

export OPENPI_DATA_HOME="/data/user_data/jasonl6/sandeep/Work/openpi"
export LEROBOT_HOME="/home/jasonl6/sandeep/Work/lq-exp/openpi/data"

uv run scripts/train.py pi0_simpler_v3_low_mem_finetune --exp-name=simpler_v3-ft --batch_size=64 --num_train_steps=40001 --resume
