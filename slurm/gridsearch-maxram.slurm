#!/bin/bash

#SBATCH --job-name gridsearch-80-20
#SBATCH --output logs/gridsearch-maxram-80-20.log
#SBATCH --error logs/gridsearch-maxram-80-20.log
#SBATCH --mail-user enrico.rossignolo@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 35-00:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 32
#SBATCH --partition allgroups
#SBATCH --mem 1000G
# uncomment to use GPU
##SBATCH --gres=gpu:rtx

set -e

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate

cd /nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier

srun python gridsearch-xgboost.py
