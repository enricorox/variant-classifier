#!/bin/bash

#SBATCH --job-name train-exact-shuffle-noestop-eta2-narrow-bm-nochr3
#SBATCH --output logs/train-exact-shuffle-noestop-eta2-narrow-bm-nochr3.log
#SBATCH --error logs/train-exact-shuffle-noestop-eta2-narrow-bm-nochr3.log
#SBATCH --mail-user enrico.rossignolo@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 35-00:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 32
#SBATCH --partition allgroups
#SBATCH --mem 200G
# uncomment to use GPU
##SBATCH --gres=gpu:rtx

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate

cd ..

srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-narrow-bm-nochr3 --shuffle_features --data main-nochr3.csv --method exact --sample_bytree 1 --iterations 1 --num_trees 100 --eta 0.2 --select regions/Br_mock_narr.csv
