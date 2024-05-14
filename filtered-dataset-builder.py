import argparse
import json
import math
import os
import time

import graphviz
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy import ndarray
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split
from xgboost import Booster


def read_feature_list(selection_file):
    features = pd.read_csv(selection_file, header=None)
    print(f"Read {len(features)} features to select")
    return features.iloc[:, 0].tolist()


def print_dataset_stats(X, y, label):
    print(f"\tData points: {X.shape[0]}")
    print(f"\t\tnumber of features: {X.shape[1]}")
    if len(y[label].unique()) == 2:
        print(f"\t\tlabel(0) counts: {(y[label] == 0).sum() / len(y[label]) * 100 : .2f} %")
        print(f"\t\tlabel(1) counts: {(y[label] == 1).sum() / len(y[label]) * 100 : .2f} %")


class DatasetBuilder:

    def __init__(self,
                 data_file,
                 selected_features_file,
                 output_file
                 ):
        self.features = None
        self.data = None
        self.data_file = data_file
        self.features_set_file = selected_features_file
        self.output_file = output_file

    def read_datasets(self, ):
        data_file = self.data_file
        selected_features_file = self.features_set_file

        start_t = time.time()
        print("Loading features...", flush=True)
        self.data = pd.read_csv(data_file, low_memory=False,
                                index_col=0,  # first column as index
                                header=0  # first row as header
                                ).astype(np.int8)
        stop_t = time.time()
        print(f"Done in {stop_t - start_t : .2f} s.", flush=True)

        if selected_features_file is not None:
            print("Selecting features...", flush=True)
            selected_features = read_feature_list(selected_features_file)
            self.data = self.data[selected_features]
            print("Done.", flush=True)

        self.features = list(self.data.columns)

        end_t = time.time()
        print(f"Total read time {end_t - start_t : .2f} s")

    def save(self):
        print(f"Saving to {self.output_file}...", flush=True)
        start_t = time.time()
        self.data.to_csv(self.output_file)
        stop_t = time.time()
        print(f"Done in {stop_t - start_t}s.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost variant classifier')

    parser.add_argument("--dataset", type=str, default="features.csv",
                        help="Features csv file")

    parser.add_argument("--select", type=str, default=None, help="List of feature to select")

    parser.add_argument("--output_file", type=str,
                        default="/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/output.csv",
                        help="Output csv file")

    args = parser.parse_args()

    print(args)

    db = DatasetBuilder(data_file=args.dataset,
                        selected_features_file=args.select,
                        output_file=args.output_file
                        )
    db.read_datasets()

    db.save()
