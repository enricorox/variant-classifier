import pandas as pd

data_file = "datasets/include-chr3/main-integers.csv"
print("Reading the datasets...")
data = pd.read_csv(data_file, low_memory=False,
                   index_col=0,  # first column as index
                   header=0  # first row as header
                   )
print("Done.")

data.head()

print("Computing constant columns...")
count = 0
constant_cols = []
for j in range(data.shape[1]):
    if j % 100000 == 0:
        print(f"Columns read: {j}")

    first = data.ilo[0, j]
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
