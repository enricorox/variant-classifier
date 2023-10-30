import lazypredict
import pandas as pd
from lazypredict.Supervised import LazyClassifier

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

data_file = "main.csv"

print(f"Using LazyPredict v{lazypredict.__version__}", flush=True)

print("Reading dataset...", flush=True)


features = pd.read_csv(data_file, nrows=0).columns[1:]
data = pd.read_csv(data_file, low_memory=False,
                   true_values=["True"],  # no inferred dtype
                   false_values=["False"],  # no inferred dtype
                   # dtype={c: int for c in features},
                   header=0,
                   index_col=0  # first column as index
                   ).astype(int)
print("Done.")

label_name = "phenotype"
train_frac = .8

X_train, X_test, y_train, y_test = train_test_split(data.drop(label_name, axis=1), data[[label_name]],
                                                    train_size=train_frac, random_state=42)

classifiers = [LinearSVC, Perceptron, RandomForestClassifier, QuadraticDiscriminantAnalysis, KNeighborsClassifier, BernoulliNB, GaussianNB, DecisionTreeClassifier]
clf = LazyClassifier(verbose=5, ignore_warnings=True, custom_metric=None, classifiers=classifiers)

print()

print("Train...", flush=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

pd.options.display.max_columns = None
pd.options.display.max_rows = None
print(models)
