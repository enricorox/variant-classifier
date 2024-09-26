#!/bin/bash

#SBATCH --job-name exact-shuffle-noestop-eta2-clusters-broads-mortality-base-int-80-20
#SBATCH --output logs/exact-shuffle-noestop-eta2-clusters-broads-mortality-base-int-80-20.log
#SBATCH --error logs/exact-shuffle-noestop-eta2-clusters-broads-mortality-base-int-80-20.log
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
dataset_ex="${base_folder}/datasets/exclude-chr3/"

active=("active-spec.csv"  "active10.csv"  "active50.csv"  "active80.csv")
control=("control-spec.csv"  "control10.csv"  "control50.csv"  "control80.csv")

set +e

for region in "${active[@]}"; do
  echo
  echo "***STARTING $region ***"
  echo
  srun python xgbclassifier.py --model_name "exact-shuffle-noestop-eta2-80-20-${region}-mortality-base-int" \
  --dataset $dataset_in/features.csv --target $dataset_in/mortality.csv \
  --features_sets_dir $dataset_in/features-sets/ --select $dataset_in/features-sets/"$region"
done

for region in "${control[@]}"; do
  echo
  echo "***STARTING $region ***"
  echo
  srun python xgbclassifier.py --model_name "exact-shuffle-noestop-eta2-80-20-${region}-mortality-base-int" \
  --dataset $dataset_in/features.csv --target $dataset_in/mortality.csv \
  --features_sets_dir $dataset_in/features-sets/ --select $dataset_in/features-sets/"$region"
done
