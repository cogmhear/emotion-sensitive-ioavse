#!/bin/bash -l
#SBATCH --gres=gpu
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=48:00:00
#SBATCH --output=../Emotion_AVSE/AVSE_AttnUNet/jobs/run_mosei_ase.out

source .bashrc
source activate py38_torch

python train.py --a_only True --emotion True --stage train --max_epochs 10 --gpu 1 --batch_size 8 --loss l1 --full_face True --model_name unet --fea_type mag

python train.py --a_only True --emotion True --stage train --max_epochs 10 --gpu 1 --batch_size 8 --loss stoi --full_face True --model_name unet --fea_type mag

python train.py --a_only True --emotion True --stage train --max_epochs 10 --gpu 1 --batch_size 8 --loss mod_loss --full_face True --model_name unet --fea_type mag
