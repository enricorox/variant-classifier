import lazypredict
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

data_file = "main.csv"

print(f"Using LazyPredict v{lazypredict.__version__}", flush=True)

print("Reading dataset...", flush=True)
data = pd.read_csv(data_file, low_memory=False,
                   true_values=["True"],  # no inferred dtype
                   false_values=["False"],  # no inferred dtype
                   index_col=0  # first column as index
                   )
print("Done.")

print(data.describe())
print(data.dtypes)


label_name = "phenotype"
train_frac = .5
point = round(len(data) * train_frac)

# shuffle
data = data.sample(frac=1.0, random_state=42)

# split
X_train, y_train = data.iloc[:point].drop(label_name, axis=1), data.iloc[:point][[label_name]]
X_test, y_test = data.iloc[point:].drop(label_name, axis=1), data.iloc[point:][[label_name]]

# X_train, y_train = X_train.to_numpy(), y_train.to_numpy()
# X_test, y_test = X_test.to_numpy(), y_test.to_numpy()

clf = LazyClassifier(verbose=5, ignore_warnings=False, custom_metric=None)
print()

print("Train...", flush=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
print(models)
