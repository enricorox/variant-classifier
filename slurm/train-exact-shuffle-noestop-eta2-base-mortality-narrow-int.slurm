#!/bin/bash

#SBATCH --job-name exact-shuffle-noestop-eta2-clusters-narrow-mortality-base-int
#SBATCH --output logs/exact-shuffle-noestop-eta2-clusters-narrow-mortality-base-int.log
#SBATCH --error logs/exact-shuffle-noestop-eta2-clusters-narrow-mortality-base-int.log
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

narrows=("Br_mock_narr.csv" "Br_NNV_narr.csv" "Hk_mock_narr.csv" "Hk_NNV_narr.csv" "narrow.csv")

set +e

for region in "${narrows[@]}"; do
  echo
  echo "***STARTING $region (cluster 0)***"
  echo
  srun python xgbclassifier.py --model_name "exact-shuffle-noestop-eta2-cluster0-${region}-mortality-base-int" \
  --dataset $dataset_in/features.csv --cluster $dataset_in/cluster-0.csv --target $dataset_in/mortality.csv \
  --features_sets_dir $dataset_in/features-sets/ --select $dataset_in/features-sets/"$region" --base_score 0
done

for region in "${narrows[@]}"; do
  echo
  echo "***STARTING $region (cluster 1)***"
  echo
  srun python xgbclassifier.py --model_name "exact-shuffle-noestop-eta2-cluster1-${region}-mortality-base-int" \
  --dataset $dataset_in/features.csv --cluster $dataset_in/cluster-1.csv --target $dataset_in/mortality.csv \
  --features_sets_dir $dataset_in/features-sets/ --select $dataset_in/features-sets/"$region" --base_score 0
done

echo "Skipping NOCHR3..."
exit

for region in "${narrows[@]}"; do
  echo
  echo "***STARTING $region (cluster 0, exclude chr3)***"
  echo
  srun python xgbclassifier.py --model_name "exact-shuffle-noestop-eta2-cluster0-nochr3-${region}-base-mortality-int" \
  --dataset $dataset_ex/features.csv --cluster $dataset_ex/cluster-0.csv --target $dataset_ex/mortality.csv \
  --features_sets_dir $dataset_ex/features-sets/ --select $dataset_ex/features-sets/"$region" --base_score 0
done

for region in "${narrows[@]}"; do
  echo
  echo "***STARTING $region (cluster 1, exclude chr3)***"
  echo
  srun python xgbclassifier.py --model_name "exact-shuffle-noestop-eta2-cluster1-nochr3-${region}-base-mortality-int" \
  --dataset $dataset_ex/features.csv --cluster $dataset_in/cluster-1.csv --target $dataset_ex/mortality.csv \
  --features_sets_dir $dataset_ex/features-sets/ --select $dataset_ex/features-sets/"$region" --base_score 0
done
