import lazypredict
import pandas as pd
from lazypredict.Supervised import LazyClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

data_file = "main.csv"

print(f"Using LazyPredict v{lazypredict.__version__}", flush=True)

print("Reading dataset...", flush=True)


def conv(v):
    if v == "True":
        return 1
    else:
        return 0


features = pd.read_csv(data_file, nrows=0).columns[1:]
data = pd.read_csv(data_file, low_memory=False,
                   # true_values=["True"],  # no inferred dtype
                   # false_values=["False"],  # no inferred dtype
                   converters={c: conv for c in features},
                   index_col=0  # first column as index
                   )
print("Done.")

label_name = "phenotype"
train_frac = .5
point = round(len(data) * train_frac)

# shuffle
data = data.sample(frac=1.0, random_state=42)

# split
X_train, y_train = data.iloc[:point].drop(label_name, axis=1), data.iloc[:point][[label_name]]
X_test, y_test = data.iloc[point:].drop(label_name, axis=1), data.iloc[point:][[label_name]]

classifiers = [LinearSVC, Perceptron, RandomForestClassifier, QuadraticDiscriminantAnalysis, KNeighborsClassifier, BernoulliNB, GaussianNB, DecisionTreeClassifier]
clf = LazyClassifier(verbose=5, ignore_warnings=True, custom_metric=None, classifiers=classifiers)

print()

print("Train...", flush=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
print(models)
