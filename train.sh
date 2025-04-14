#!/bin/bash
#SBATCH --job-name=oasis         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=28        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=0:59:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/oasis-%J.log
#SBATCH --error=logs/oasis-%J.log

# Retrieve the passed WORKDIR
echo "Current working directory: $(pwd)"
export WORKDIR=game_video_world_model
echo "Received WORKDIR: $WORKDIR"
export PYTHONPATH=$WORKDIR  # add workdir to PYTHONPATH to ensure modules can be found
#export HYDRA_FULL_ERROR=1

# load modules or conda environments here
source ~/.bashrc
#eval "$(conda shell.bash hook)"  # this is needed to load python packages correctly

# Activate Python virtual environment
echo "Contents of the current directory:"
ls -lah

source game_video_world_model/oasis/bin/activate

echo "Using Python from: $(which python3)"
python3 --version

# python train_oasis/main.py
# torchrun --nproc_per_node=8 train_oasis/main.py
torchrun --nproc_per_node=8 $WORKDIR/train_oasis/main.py

# rm -rf $WORKDIR
