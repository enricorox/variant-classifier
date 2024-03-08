#!/bin/bash

#SBATCH --job-name dataset-stats
#SBATCH --output logs/dataset-stats.log
#SBATCH --error logs/dataset-stats.log
#SBATCH --mail-user enrico.rossignolo@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 35-00:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 1
#SBATCH --partition allgroups
#SBATCH --mem 64G
# uncomment to use GPU
##SBATCH --gres=gpu:rtx

set -e

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate

cd /nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/

srun python dataset-stats.py
