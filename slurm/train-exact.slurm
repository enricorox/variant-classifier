#!/bin/bash

#SBATCH --job-name train-exact
#SBATCH --output logs/train-exact.log
#SBATCH --error logs/train-exact.log
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

srun python XGBoostVariant.py --model_name exact --csv main.csv --method exact --sample_bytree 1 --iterations 1 --early_stopping 50
