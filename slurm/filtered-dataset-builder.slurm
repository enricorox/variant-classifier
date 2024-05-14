#!/bin/bash

#SBATCH --job-name filtered-dataset-builder
#SBATCH --output logs/filtered-dataset-builder.log
#SBATCH --error logs/filtered-dataset-builder.log
#SBATCH --mail-user enrico.rossignolo@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 35-00:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 1
#SBATCH --partition allgroups
#SBATCH --mem 200G
# uncomment to use GPU
##SBATCH --gres=gpu:rtx

set -e

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate

base_folder="/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier"

cd $base_folder

dataset_in="${base_folder}/datasets/include-chr3/"
# dataset_ex="${base_folder}/datasets/exclude-chr3/"

set +e

echo
echo "***Building filtered dataset***"
echo
srun python filtered-dataset-builder.py --dataset $dataset_in/features.csv --select $dataset_in/features-sets/"Hk_NNV_broad.csv" --output_file $dataset_in/"features-Hk_NNV_broad.csv"
