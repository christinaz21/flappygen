#!/bin/bash
#SBATCH --job-name=oasis         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=4               # total number of tasks across all nodes
# SBATCH --cpus-per-task=48        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=23:59:59          # total run time limit (HH:MM:SS)
#SBATCH --output=logs/oasis-%J.log
#SBATCH --error=logs/oasis-%J.log

# export DS_BUILD_OPS=0
# export DS_BUILD_CPU_ADAM=0
# export DS_BUILD_UTILS=0
# export DS_BUILD_TRANSFORMER=0
# export TORCH_CUDA_ARCH_LIST="8.0 8.6 9.0"
# export TRITON_CACHE_DIR="/tmp/triton_cache"

# export CUDA_HOME=/usr/local/cuda-12.8
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
# export CPATH=$CUDA_HOME/include:$CPATH
# export LDFLAGS="-L$CUDA_HOME/lib64"
# export DS_BUILD_OPTS="-L/usr/local/cuda-12.8/lib64 -lcurand"

# Retrieve the passed WORKDIR
echo "Current working directory: $(pwd)"
#export WORKDIR=game_video_world_model
#echo "Received WORKDIR: $WORKDIR"
#export PYTHONPATH=$WORKDIR  # add workdir to PYTHONPATH to ensure modules can be found
#export HYDRA_FULL_ERROR=1

# load modules or conda environments here
source ~/.bashrc
eval "$(conda shell.bash hook)"  # this is needed to load python packages correctly

# Activate Python virtual environment
echo "Contents of the current directory:"
ls -lah

# source oasis/bin/activate
conda activate /scratch/gpfs/cz5047/oasis-conda

echo "Environment activated: $(conda info --envs)"

echo "Using Python from: $(which python)"
python --version

echo "Using torchrun from: $(which torchrun)"
# export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
# export MASTER_PORT=29500

# python train_oasis/main.py
# torchrun --nproc_per_node=8 train_oasis/main.py
 torchrun --nnodes=1 --nproc_per_node=4 train_oasis/main.py

# torchrun \
#     --nnodes=$SLURM_NNODES \
#     --nproc_per_node=$(nvidia-smi -L | wc -l) \
#     --rdzv_id=$SLURM_JOB_ID \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     train_oasis/main.py

# rm -rf $WORKDIR
