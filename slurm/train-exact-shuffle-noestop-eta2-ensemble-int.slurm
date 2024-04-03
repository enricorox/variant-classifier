#!/bin/bash

#SBATCH --job-name train-exact-shuffle-noestop-eta2-ensemble-int
#SBATCH --output logs/train-exact-shuffle-noestop-eta2-ensemble-int.log
#SBATCH --error logs/train-exact-shuffle-noestop-eta2-ensemble-int.log
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

set -e

source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate

base_folder="/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier"

cd $base_folder

dataset_in="${base_folder}/datasets/include-chr3/"
dataset_ex="${base_folder}/datasets/exclude-chr3/"

  srun python xgbclassifier.py --model_name "train-exact-shuffle-noestop-eta2-ensemble-int" \
    --dataset "${dataset_ex}/features.csv" --target "${dataset_ex}/mortality.csv" --select "${dataset_ex}/regions/ensemble-features.csv" \
    --features_sets_dir "${dataset_in}/regions/"
