#!/bin/bash
#SBATCH --job-name=bc
#SBATCH --output=logs/out/DALIA_FSDP_sAC_72B_%j.out 
#SBATCH --error=logs/err/DALIA_FSDP_sAC_72B_%j.err
#SBATCH --gpus-per-node=4
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4 
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=36 

## load module 
module purge
module load slurm/slurm/24.11

## cache dir
export HF_DATASETS_CACHE="/lustre/work/sos/ssos040/cache_hf"


## Distribution setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

export NCCL_DEBUG=WARN

export APPTAINER_BINDPATH="$WORK,$ALL_WORK"

ulimit -s 8192

## launch script on every task 
set -x
time srun apptainer exec --nv \
    $WORK/nemo2506_arm.sif \
    python $MYWORK/FSDP_sAC.py --test --model Qwen/Qwen2.5-72B-Instruct --sac 1/2 --bsz 2 --grad-acc 1 --cpu-usage --compile
date
