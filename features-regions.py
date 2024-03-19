import os
import sys

import pandas as pd

if len(sys.argv) == 2:
    os.chdir(sys.argv[1])

# List of CSV file paths
regions_files = ["Br_mock_broad.csv", "Br_NNV_broad.csv", "Hk_mock_broad.csv", "Hk_NNV_broad.csv", "broad.csv",
                 "Br_mock_narr.csv", "Br_NNV_narr.csv", "Hk_mock_narr.csv", "Hk_NNV_narr.csv", "narrow.csv"]
counts_file = "importance-0.counts.csv"
gains_file = "importance-0.gains.csv"
data_ensemble_file = "data_ensemble.csv"

# regions_files = ["broad.csv"]
# features_file = "regions/Hk_NNV_broad.csv"

# Function to read CSV file and return a list of strings from a specific column
def read_csv_and_get_list(file_path, column_idx):
    df = pd.read_csv(file_path, header=None)
    return df.iloc[:, column_idx].tolist()


# Specify the column name containing strings
column_idx = 0

# Create a list to store lists of strings from each CSV file
gains = pd.read_csv(gains_file, index_col=0, header=None)
total_gain = gains.loc[:, 1].sum()

counts = pd.read_csv(counts_file, index_col=0, header=None)
total_counts = counts.loc[:, 1].sum()
features = set(counts.index)

# Read from each CSV file and append the list to the main list
print("Peaks\tCounts\tWeights\tGains")
for file in regions_files:
    current_list = read_csv_and_get_list("regions/" + file, column_idx)
    common_strings = features.intersection(current_list)

    grouped_gains = gains.loc[list(common_strings), 1]
    grouped_counts = counts.loc[list(common_strings), 1]
    print(f"{file}\t"
          f"{len(common_strings) / len(features) * 100 : .2f} %\t"
          f"{grouped_gains.sum() / total_gain * 100 : .2f} %\t"
          f"{grouped_counts.sum() / total_counts * 100 : .2f}%")
    if file == "broad.csv":
        print()

data_ensemble = pd.read_csv(data_ensemble_file, index_col=0, header=0)
ensemble_features = list(features.intersection(data_ensemble.index.values))
data_ensemble = data_ensemble.loc[ensemble_features, :]
data_ensemble["gain"] = gains.loc[ensemble_features, :]
data_ensemble.sort_values(by="gain")
data_ensemble.to_csv("ensemble.csv")

print(f"#features in the ensemble: {len(ensemble_features)}")

if len(ensemble_features) > 0:
    info_funct = data_ensemble["funct"].value_counts()
    info_n_tissue = data_ensemble["n_tissue"].value_counts()

    print(info_funct)
    print(info_n_tissue)
else:
    print("No features in the data ensemble!!!")
