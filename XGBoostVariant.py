from collections import defaultdict

import graphviz
import pandas as pd
import xgboost as xgb
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from xgboost import Booster


class XGBoostVariant:
    bst: Booster
    num_trees: int
    best_it: int
    best_score: float

    dtrain: xgb.DMatrix
    dvalidation: xgb.DMatrix
    dtest: xgb.DMatrix

    y_pred: ndarray
    y_test: ndarray

    def __init__(self, model_name="xgbtree", num_trees=10):
        self.random_state = 42
        self.model_name = model_name
        self.num_trees = num_trees
        self.label = "phenotype"
        self.train_frac = .8

        print(f"Using XGBoost version {xgb.__version__}")

    def read_datasets(self, data_file, feature_weights=None):
        print("Loading data...", flush=True)
        data = pd.read_csv(data_file, low_memory=False,
                           true_values=["True"],  # no inferred dtype
                           false_values=["False"],  # no inferred dtype
                           index_col=0  # first column as index
                           )

        # shuffle
        data = data.sample(frac=1.0, random_state=self.random_state)

        point = round(len(data) * self.train_frac)
        X_train, y_train = data.iloc[:point].drop(self.label, axis=1), data.iloc[:point][[self.label]]
        X_test, y_test = data.iloc[point:].drop(self.label, axis=1), data.iloc[point:][[self.label]]

        print("Stats (train data):")
        print(f"\tData points: {X_train.shape[0]}")
        print(f"\t\tnumber of features: {X_train.shape[1]}")
        print(f"\t\tlabel(0) counts: {(y_train[self.label] == 0).sum() / len(y_train[self.label]) * 100 : .2f} %")
        print(f"\t\tlabel(1) counts: {(y_train[self.label] == 1).sum() / len(y_train[self.label]) * 100 : .2f} %")
        print("Transforming into DMatrices...")
        self.dtrain = xgb.DMatrix(X_train, y_train)

        print()

        print("Stats (test data):")
        print(f"\tData points: {X_test.shape[0]}")
        print(f"\t\tnumber of features: {X_train.shape[1]}")
        print(f"\t\tlabel(0) counts: {(y_test[self.label] == 0).sum() / len(y_test[self.label]) * 100 : .2f} %")
        print(f"\t\tlabel(1) counts: {(y_test[self.label] == 1).sum() / len(y_test[self.label]) * 100 : .2f} %")
        print("Transforming into DMatrices...")
        self.y_test = y_test
        self.dtest = xgb.DMatrix(X_test)

        print()

        self.dvalidation = None

        if feature_weights is not None:
            assert len(feature_weights) == self.dtrain.num_col()
            self.dtrain.set_info(feature_weights=feature_weights)

    def fit(self, params=None, evals=None):
        if self.dtrain is None:
            raise Exception("Need to load training datasets first!")

        if params is None:
            params = {"verbosity": 1, "device": "cpu", "objective": "binary:hinge", "tree_method": "hist",
                      "colsample_bytree": .8, "seed": self.random_state}
            # params["eval_metric"] = "auc"
        if evals is None:
            if self.dvalidation is None:
                evals = [(self.dtrain, "training")]
            else:
                evals = [(self.dtrain, "training"), (self.dvalidation, "validation")]

        self.bst = xgb.train(params=params, dtrain=self.dtrain,
                             num_boost_round=self.num_trees,
                             evals=evals,
                             verbose_eval=10,
                             early_stopping_rounds=50,
                             )

        # update number of trees in case of early stopping
        self.num_trees = self.bst.num_boosted_rounds()
        self.best_it = self.bst.best_iteration
        self.best_score = self.bst.best_score

        # save model
        self.bst.save_model(f"{self.model_name}.json")

    def predict(self, iteration_range=None):
        if iteration_range is None:
            iteration_range = (0, self.best_it)

        self.y_pred = self.bst.predict(self.dtest, iteration_range=iteration_range)

    def print_stats(self):
        print("\nPrediction stats:")

        print(f"Best score: {self.best_score}")
        print(f"Best iteration: {self.best_it}")

        conf_mat = confusion_matrix(self.y_test, self.y_pred)
        true_neg = conf_mat[0][0]
        true_pos = conf_mat[1][1]
        false_neg = conf_mat[1][0]
        false_pos = conf_mat[0][1]

        assert (true_pos + false_neg) == sum(self.y_test[self.label])
        assert (true_neg + false_pos) == len(self.y_test[self.label]) - sum(self.y_test[self.label])
        assert (true_neg + true_pos + false_neg + false_pos) == len(self.y_test[self.label])

        print(f"TN={true_neg}\tFP={false_pos}")
        print(f"FN={false_neg}\tTP={true_pos}")

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        print(f"Accuracy = {accuracy * 100 : .2f} %")

        importance = self.bst.get_score(importance_type="weight")
        importance = sorted(importance.items(), key=lambda item: item[1], reverse=True)
        print("Top 10 features:")
        print(importance[:100])

    def plot_trees(self, tree_set=None):
        print("Printing trees...")
        if tree_set is None:
            tree_set = range(self.num_trees)

        for i in tree_set:
            graph: graphviz.Source
            graph = xgb.to_graphviz(self.bst, num_trees=i)
            graph.render(filename=f"{self.model_name}-{i}", directory="trees", format="png", cleanup=True)
        print("Done.")


if __name__ == "__main__":
    dataset_folder = "./"
    data_file = dataset_folder + "main.csv"

    clf = XGBoostVariant(num_trees=100)
    clf.read_datasets(data_file)
    clf.fit()
    clf.predict()
    clf.print_stats()
    clf.plot_trees()

# TODO add max_depth
# TODO add variable constraints
# TODO add features sampling by *
# TODO add data points sampling
# TODO add scale_pos_weight to balance classes
# TODO add eta (learning rate)
# TODO add gamma (high for conservative algorithm)
