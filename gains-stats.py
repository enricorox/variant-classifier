import glob

import pandas as pd

paths = ["./results/all/", "./results/peaks/", "./results/peaks-nochr3/"]
csv_out = "all-gains.csv"
k = 10
counts = {}
gains = {}

for path in paths:
    counts = {}
    gains = {}
    norm_gains = {}
    for file in glob.glob(path + "exact-shuffle-*/importance-0.gains.csv"):
        print(f"Reading file {file}...")
        df = pd.read_csv(file, index_col=0, names=["gains"])
        for e in df.index:
            counts[e] = counts.get(e, 0) + 1
            gains[e] = gains.get(e, 0) + df.loc[e, "gains"]
            norm_gains[e] = gains[e] / counts[e]

    csv_out = path + "all-gains.csv"

    with open(csv_out, "w") as csv:
        csv.write("SNIP, counts, gains\n")
        for e in counts.keys():
            csv.write(f"{e}, {counts[e]}, {gains[e]}, {norm_gains[e]}\n")

    print(f"*** Output written to {csv_out}")

    print(f"Top {k} - counts")
    print("feature, count, gain, normalized gain")
    for e in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:k]:
        key = e[0]
        print(f"{key}, {counts[key]}, {gains[key]}, {norm_gains[key]}")

    print(f"Top {k} - gains")
    print("feature, count, gain, normalized gain")
    for e in sorted(gains.items(), key=lambda item: item[1], reverse=True)[:k]:
        key = e[0]
        print(f"{key}, {counts[key]}, {gains[key]}, {norm_gains[key]}")

    print(f"Top {k} - normalized gains")
    print("feature, count, gain, normalized gain")
    for e in sorted(norm_gains.items(), key=lambda item: item[1], reverse=True)[:k]:
        key = e[0]
        print(f"{key}, {counts[key]}, {gains[key]}, {norm_gains[key]}")