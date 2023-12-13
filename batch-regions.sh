#!/bin/bash
source /nfsd/bcb/bcbg/rossigno/.miniconda3/bin/activate

for dir in train-exact-shuffle-noestop-eta2-*; do
  (cd $dir
  echo "Analyzing model $dir"
  python features-regions.py
  )
done