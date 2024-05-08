#!/bin/bash

#SBATCH --job-name train-random-forest-hk-nnv-pipeline
#SBATCH --output logs/train-random-forest-hk-nnv-pipeline.log
#SBATCH --error logs/train-random-forest-hk-nnv-pipeline.log
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
echo "***STARTING RF (cluster 0)***"
echo
srun python xgbclassifier.py --model_name "train-random-forest-hk-nnv-pipeline-rf" --method hist \
--dataset $dataset_in/features.csv --target $dataset_in/mortality.csv --cluster $dataset_in/cluster-0.csv --select $dataset_in/features-sets/"Hk_NNV_broad.csv"\
--num_trees 1 --num_parallel_trees 1000 --max_depth 100 \
--sample_bynode 0.001 --subsample 0.99

features_dir=$(realpath "train-random-forest-hk-nnv-pipeline-rf")
features_gains=$features_dir/importance-0.gains.csv
features=$features_dir/extracted_features.csv
awk -F"," '{print $1}' "$features_gains" > "$features"

echo
echo "***STARTING XGB (cluster 0)***"
echo
srun python xgbclassifier.py --model_name "train-random-forest-hk-nnv-pipeline-xgb" --method hist \
--dataset $dataset_in/features.csv --target $dataset_in/mortality.csv --cluster $dataset_in/cluster-0.csv --select "$features"\
--num_trees 10 --num_parallel_trees 1000 --max_depth 100 \
--sample_bynode 0.001 --subsample 0.99