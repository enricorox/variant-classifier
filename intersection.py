# Find the intersection of all lists
import pandas as pd

peaks_files = ["Br_mock_broad.csv", "Br_NNV_broad.csv", "Hk_mock_broad.csv", "Hk_NNV_broad.csv", "broad.csv",
                 "Br_mock_narr.csv", "Br_NNV_narr.csv", "Hk_mock_narr.csv", "Hk_NNV_narr.csv", "narrow.csv"]
# peaks_files = ["broad.csv", "Hk_NNV_broad.csv"]
print("Reading files...")
lists_of_strings = []
for pf in peaks_files:
    strings = pd.read_csv("regions/" + pf, header=None)
    strings = strings.iloc[:, 0].tolist()
    lists_of_strings.append(strings)

print("Computing intersection...", flush=True)
common_strings = set(lists_of_strings[0]).intersection(*lists_of_strings[1:])

# Print or use the common strings
print(f"Common Strings: {len(common_strings)}")
with open("common.csv", "a") as f:
    for s in common_strings:
        f.write(f"{s}\n")
