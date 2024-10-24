#!/bin/bash
#SBATCH --mem=20g
#SBATCH --time=14:00:00
#SBATCH --partition=gpu
#SBATCH --gres gpu:quadro:1
module load StdEnv/2020
module load cuda
module load python/3.8.10
python script2.py

