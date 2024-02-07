#!/bin/bash
#SBATCH --array=1-1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-core=1
#SBATCH --time=12:00:00
#SBATCH --job-name=_warmup
#SBATCH --error=err/%A_%a.err
#SBATCH --output=out/%A_%a.out
####SBATCH --mem=10000
#SBATCH --gpus=a100:1
####SBATCH --gpus=a100_1g.5gb:1
#SBATCH --partition=gpu
####SBATCH --partition=scavenger

cd $SLURM_SUBMIT_DIR

srun time python3 train_wide_resnet_linear_warmup.py --dataset cifar10 --loss_name xent --abc sp --width 16 --widening_factor 4 --depth 16 --scale 0.0 --varw 2.0 --act relu --lr_exp_start 5.50 --warm_steps 512 --num_epochs 200 --init_seed 1 --momentum 0.0 --augment False  
