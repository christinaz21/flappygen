#!/bin/bash
#SBATCH --job-name=flappy_train
#SBATCH --output=logs/%x_%j.out        # Logs go to logs/flappy_train_JOBID.out
#SBATCH --error=logs/%x_%j.err         # Error logs
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --mem=32G                      # RAM
#SBATCH --cpus-per-task=4              # Number of CPU cores
#SBATCH --time=59:00                # Max time (8 hours)

# Activate your conda environment
source ~/.bashrc
conda activate flappy-env

# Make sure logs dir exists
mkdir -p logs

# Run the training script
python train_flappy.py
