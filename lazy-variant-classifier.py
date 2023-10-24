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
y = data["phenotype"].values
X = data.drop(columns=["phenotype"]).values
print("Done.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=123)
clf = LazyClassifier(verbose=5, ignore_warnings=True, custom_metric=None)

print()

print("Train...", flush=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
print(models)
