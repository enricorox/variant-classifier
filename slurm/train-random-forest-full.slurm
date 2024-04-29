#!/bin/bash

#SBATCH --job-name train-random-forest-full.slurm
#SBATCH --output logs/train-random-forest-full.slurm.log
#SBATCH --error logs/train-random-forest-full.slurm.log
#SBATCH --mail-user enrico.rossignolo@studenti.unipd.it
#SBATCH --mail-type ALL
#SBATCH
#SBATCH --time 35-00:00:00
# use many CPU if not using CUDA
#SBATCH --cpus-per-task 24
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
echo "***STARTING RF 80/20***"
echo
srun python xgbclassifier.py --model_name "train-random-forest-full.slurm" --method hist \
--dataset $dataset_in/features.csv --target $dataset_in/mortality.csv \
--num_trees 1 --num_parallel_trees 1000 --sample_bynode 0.001 \
--max_depth 2

