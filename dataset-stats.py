import numpy as np
import pandas as pd

data_file = "datasets/include-chr3/main-integers.csv"
new_labels_file = "datasets/download/EBV.txt"

print("Reading the datasets...", flush=True)
data = pd.read_csv(data_file, low_memory=False,
                   index_col=0,  # first column as index
                   header=0  # first row as header
                   )

new_label = pd.read_csv(new_labels_file, low_memory=False,
                        delimiter=" ",
                        index_col=0,  # first column as index
                        header=0  # first row as header
                        )

print("Done.", flush=True)

print(data)
print(new_label)


def find_constant_cols():
    print("Computing constant columns...")
    count = 0
    constant_cols = []
    for j in range(data.shape[1]):
        if j % 100000 == 0:
            print(f"Columns read: {j}")

        first = data.iloc[0, j]
        is_constant_col = True
        for i in range(1, data.shape[0]):
            if first != data.iloc[i, j]:
                is_constant_col = False
                break
        if is_constant_col:
            count += 1
            print(f"Constant cols: {constant_cols}")
            constant_cols.append(data.columns[j])

    pd.DataFrame(constant_cols).to_csv("constant_columns.csv")


# NB: no constant cols found!
# find_constant_cols()

mortality = data["phenotype"].astype(np.int8)
clusters = data["cluster"].astype(np.int8)
ebv = new_label["ebv"].astype(np.float16)
data.drop(columns=["phenotype", "cluster"], inplace=True)
data = data.astype(np.int8)

print("Writing labels to HDF5...", flush=True)
mortality.to_hdf("dataset.hdf5", key="mortalities")
clusters.to_hdf("dataset.hdf5", key="clusters")
ebv.to_hdf("dataset.hdf5", key="ebv")

print("Writing features to HDF5...", flush=True)
data.to_hdf("datasets.hdf5", key="features")
