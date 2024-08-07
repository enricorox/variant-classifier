#!/bin/bash

#SBATCH --job-name train-exact-shuffle-noestop-eta2-ALL-int
#SBATCH --output logs/train-exact-shuffle-noestop-eta2-ALL-int.log
#SBATCH --error logs/train-exact-shuffle-noestop-eta2-ALL-int.log
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

regions=("Br_mock_broad.csv" "Br_NNV_broad.csv" "Hk_mock_broad.csv" "Hk_NNV_broad.csv" "broad.csv"
               "Br_mock_narr.csv" "Br_NNV_narr.csv" "Hk_mock_narr.csv" "Hk_NNV_narr.csv" "narrow.csv")



for region in "${regions[@]}"; do
  echo
  echo "***STARTING $region (INCLUDE CHR3)***"
  echo
  srun python xgbclassifier.py --model_name "exact-shuffle-noestop-eta2-${region}-int" \
    --dataset "${dataset_in}/features.csv" --target "${dataset_in}/mortality.csv" --select "${dataset_in}/features-sets/${region}"\
    --features_sets_dir "${dataset_in}/features-sets/"
done

for region in "${regions[@]}"; do
  echo
  echo "***STARTING $region (EXCLUDE CHR3)***"
  echo
  srun python xgbclassifier.py --model_name "exact-shuffle-noestop-eta2-nochr3-${region}-int" \
    --dataset "${dataset_ex}/features.csv" --target "${dataset_ex}/mortality.csv" --select "${dataset_ex}/features-sets/${region}" \
    --features_sets_dir "${dataset_ex}/features-sets/"
done