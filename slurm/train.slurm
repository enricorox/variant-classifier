#!/bin/bash

#SBATCH --job-name train-xgboost
#SBATCH --output logs/train.log
#SBATCH --error logs/train.log
#SBATCH --mail-user enrico.rossignolo@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 35-00:00:00
#SBATCH --cpus-per-task 1
#SBATCH --partition allgroups
#SBATCH --mem 500G
#SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate

cd ..

# srun python dna-classifier.py
srun python XGBoostVariant.py
