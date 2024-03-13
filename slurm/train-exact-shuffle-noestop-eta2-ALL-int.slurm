#!/bin/bash

#SBATCH --job-name train-exact-shuffle-noestop-eta2-ALL
#SBATCH --output logs/train-exact-shuffle-noestop-eta2-ALL.log
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

cd /nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier

echo
echo "***STARTING Br_NNV_broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/Br_NNV_broad.csv

echo
echo "***STARTING Br_NNV_narr.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/Br_NNV_narr.csv

echo
echo "***STARTING Br_mock_broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/Br_mock_broad.csv

echo
echo "***STARTING Br_mock_narr.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/Br_mock_narr.csv

echo
echo "***STARTING Hk_NNV_broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/Hk_NNV_broad.csv

echo
echo "***STARTING Hk_NNV_narr.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/Hk_NNV_narr.csv

echo
echo "***STARTING Hk_mock_broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/Hk_mock_broad.csv

echo
echo "***STARTING Hk_mock_narr.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/Hk_mock_narr.csv

echo
echo "***STARTING broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/broad.csv

echo
echo "***STARTING narrow.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/exclude-chr3/ --select regions/narrow.csv



echo
echo "***STARTING Br_NNV_broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/Br_NNV_broad.csv

echo
echo "***STARTING Br_NNV_narr.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/Br_NNV_narr.csv

echo
echo "***STARTING Br_mock_broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/Br_mock_broad.csv

echo
echo "***STARTING Br_mock_narr.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/Br_mock_narr.csv

echo
echo "***STARTING Hk_NNV_broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/Hk_NNV_broad.csv

echo
echo "***STARTING Hk_NNV_narr.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/Hk_NNV_narr.csv

echo
echo "***STARTING Hk_mock_broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/Hk_mock_broad.csv

echo
echo "***STARTING Hk_mock_narr.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/Hk_mock_narr.csv

echo
echo "***STARTING broad.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/broad.csv

echo
echo "***STARTING narrow.csv***"
echo
srun python XGBoostVariant.py --model_name exact-shuffle-noestop-eta2-nochr3-int --dataset datasets/include-chr3/ --select regions/narrow.csv
