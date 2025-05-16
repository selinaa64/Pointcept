#!/bin/bash
#SBATCH --partition=gpu1
#SBATCH --ntasks-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=semoit00@hs-esslingen.de
#SBATCH --job-name=pointTransformerV3_train
#SBATCH --output=slurm_%j.out



module load devel/miniforge

conda activate pointcept-torch2.5.0-cu12.4
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

sh scripts/train.sh -p python -d modelnet40 -c cls-ptv3-v1m1-0-base -n cls-ptv3-v1m1-0-base