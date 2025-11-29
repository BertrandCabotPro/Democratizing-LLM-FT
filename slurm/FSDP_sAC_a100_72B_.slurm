#!/bin/bash
#SBATCH --job-name=bc
#SBATCH --output=logs/out/FSDP_sAC_a100_72B_%j.out 
#SBATCH --error=logs/err/FSDP_sAC_a100_72B_%j.err
#SBATCH --gres=gpu:8
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --hint=nomultithread 
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH -C a100
#SBATCH --partition=gpu_p5
#SBATCH --account=sos@a100


## load module 
module purge
module load arch/a100
#module load pytorch-gpu/py3/2.8.0
module load singularity

## cache dir
export HF_DATASETS_CACHE="/lustre/fsn1/projects/idris/sos/ssos040/HF_CACHE/tulu3/a100"

export SINGULARITY_BINDPATH="$WORK,$ALL_CCFRWORK,$DSDIR,$SCRATCH,$ALL_CCFRSCRATCH,$JOBSCRATCH,/lustre/fswork/dataset"
## Distribution setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

# detect the IB trouble
export NCCL_DEBUG=WARN


## launch script on every task 
set -x
#time srun python -u FSDP_sAC.py --test --model Qwen/Qwen2.5-72B-Instruct --sac 7/8 --bsz 2 --grad-acc 1 --compile
time srun singularity exec --nv --bind $SINGULARITY_BINDPATH \
    $SINGULARITY_ALLOWED_DIR/nemo2506_amd.sif \
    python FSDP_sAC.py --test --model Qwen/Qwen2.5-72B-Instruct --sac 1 --bsz 2 --grad-acc 1 --cpu-usage --compile

date
