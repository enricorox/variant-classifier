#!/bin/bash

#SBATCH --job-name train-xgboost-ch3
#SBATCH --output logs/train-ch3.log
#SBATCH --error logs/train-ch3.log
#SBATCH --mail-user enrico.rossignolo@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 35-00:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 32
#SBATCH --partition allgroups
#SBATCH --mem 500G
# uncomment to use GPU
##SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate

cd ..

# srun python dna-classifier.py
srun python XGBoostVariant.py
