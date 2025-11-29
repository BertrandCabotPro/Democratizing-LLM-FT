#!/bin/bash
#SBATCH --job-name=bc
#SBATCH --output=logs/out/DALIA_NeMo_FSDP_TP_72B_%j.out 
#SBATCH --error=logs/err/DALIA_NeMo_FSDP_TP_72B_%j.err
#SBATCH --gpus-per-node=4
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4 
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=36 

## load module 
module purge
module load slurm/slurm/24.11

## cache dir
export HF_DATASETS_CACHE="/lustre/work/sos/ssos040/cache_hf"


## Distribution setup
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

export NCCL_DEBUG=WARN

export APPTAINER_BINDPATH="$WORK,$ALL_WORK"
export MYPATH=/lustre/work/sos/commun/BC/LLMSFTComBenchmarking/InstructFT


## launch script on every task 
set -x
time srun apptainer exec --nv \
    $WORK/nemo2506_arm.sif \
    python $MYPATH/nemo_TPFSDP2.py --model Qwen/Qwen2.5-72B-Instruct \
                             --use-te-optimizer \
                             --strategy fsdp2 \
                             --devices 4 \
                             --num-nodes 16 \
                             --tp-size 4 \
                             --cp-size 1 \
                             --batch-size 8 \
                             --enable-grad-ckpt \

date
