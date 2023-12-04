import pandas as pd

# List of CSV file paths
regions_files = ["Br_mock_broad.csv", "Br_NNV_broad.csv", "Hk_mock_broad.csv", "Hk_NNV_broad.csv", "broad.csv", "Br_mock_narr.csv",  "Br_NNV_narr.csv",   "Hk_mock_narr.csv", "Hk_NNV_narr.csv", "narrow.csv"]
features_file = "importance-0.counts.csv"


# Function to read CSV file and return a list of strings from a specific column
def read_csv_and_get_list(file_path, column_idx):
    df = pd.read_csv(file_path)
    return df.iloc[:, column_idx].tolist()


# Specify the column name containing strings
column_idx = 0

# Create a list to store lists of strings from each CSV file
features = set(read_csv_and_get_list(features_file, column_idx))

# Read from each CSV file and append the list to the main list
print("Intersection with:")
for file in regions_files:
    # print(f"Reading {file}...", flush=True)
    current_list = read_csv_and_get_list("regions/"+file, column_idx)
    common_strings = features.intersection(current_list)
    print(f"\t{file},\t{len(common_strings) / len(features) * 100 : .2f} %")

"""# Find the intersection of all lists
print("Computing intersection...", flush=True)
common_strings = set(lists_of_strings[0]).intersection(*lists_of_strings[1:])

# Print or use the common strings
print(f"Common Strings: {len(common_strings)}")
with open("common.csv", "a") as f:
    for s in common_strings:
        f.write(f"{s}\n")
"""