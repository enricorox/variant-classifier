import glob

import pandas as pd

path = "./"

counts = {}
gains = {}

for file in glob.glob(path + "train-*/importance-0.gains.csv"):
    print(f"Reading file {file}...")
    df = pd.read_csv(file, index_col=0, usecols=["gains"])
    for e in df.index:
        counts[e] = counts.get(e, 0) + 1
        gains[e] = gains.get(e, 0) + df[e, "gains"]

with open("all-gains.csv", "w") as csv:
    csv.write("SNIP, counts, gains")
    for e in counts.keys():
        csv.write(f"{e}, {counts[e]}, {gains[e]}")
