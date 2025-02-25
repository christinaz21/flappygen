#!/bin/bash
#SBATCH --job-name=oasis         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=90        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=480G
#SBATCH --gres=gpu:8             # number of gpus per node
#SBATCH --time=144:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/oasis-%J.log
#SBATCH --error=logs/oasis-%J.log


# Retrieve the passed WORKDIR
echo "Received WORKDIR: $WORKDIR"
export PYTHONPATH=$WORKDIR  # add workdir to PYTHONPATH to ensure modules can be found

# load modules or conda environments here
source ~/.bashrc
eval "$(conda shell.bash hook)"  # this is needed to load python packages correctly

# active conda environment
source activate oasis

echo "Environment activated: $(conda info --envs)"
which python
python --version

# python train_oasis/main.py
# torchrun --nproc_per_node=8 train_oasis/main.py
torchrun --nproc_per_node=8 $WORKDIR/train_oasis/main.py

# rm -rf $WORKDIR
