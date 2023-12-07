import argparse
import os
import time

import graphviz
import pandas as pd
import xgboost as xgb
from numpy import ndarray
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import Booster
import pyarrow


def shuffle_columns(df, seed=42):
    # print(df)
    # Get the last column
    last_column = df.columns[-1]

    # Select all but the last columns and shuffle them
    columns_to_shuffle = df.columns[:-1]
    shuffled_df = df[columns_to_shuffle].sample(frac=1, axis=1, random_state=seed)

    # Add the last column back to the shuffled DataFrame
    shuffled_df[last_column] = df[last_column]
    # print(shuffled_df)
    return shuffled_df


def read_feature_list_parquet(selection_file):
    if selection_file is None:
        return None
    features = pd.read_csv(selection_file, header=None)
    print(f"Read {len(features)} features to select")
    features_to_extract = []
    for feature in features.iloc[:, 0]:
        features_to_extract.append(feature.replace(".", "_"))
    # return features.iloc[:, 0]
    return features_to_extract


def read_feature_list(selection_file):
    if selection_file is None:
        return None
    features = pd.read_csv(selection_file, header=None)
    print(f"Read {len(features)} features to select")
    return features.iloc[:, 0]


def print_stats(X, y, label):
    print(f"\tData points: {X.shape[0]}")
    print(f"\t\tnumber of features: {X.shape[1]}")
    print(f"\t\tlabel(0) counts: {(y[label] == 0).sum() / len(y[label]) * 100 : .2f} %")
    print(f"\t\tlabel(1) counts: {(y[label] == 1).sum() / len(y[label]) * 100 : .2f} %")


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

    features = None
    importance_counts = None
    importance_gain = None

    def __init__(self, model_name="default-model", num_trees=10, max_depth=6, eta=.3, early_stopping=50,
                 sample_bytree=250 / 6072853, method="hist"):
        self.max_depth = max_depth
        self.eta = eta
        self.early_stopping = early_stopping
        self.by_tree = sample_bytree
        self.random_state = 42
        self.model_name = model_name
        self.num_trees = num_trees
        self.label_name = "phenotype"
        self.train_frac = .8
        self.method = method

        if early_stopping is None:
            self.early_stopping = self.num_trees

        print(f"Using XGBoost version {xgb.__version__}")

    def read_datasets(self, data_file, validation=False, feature_weights=None, shuffle_features=False, select=None):
        start_t = time.time()
        print("Loading data...", flush=True)
        if "csv" in data_file:
            data = pd.read_csv(data_file, low_memory=False,
                               # usecols=lambda c: select is None or c in read_feature_list(select),
                               true_values=["True"],  # no inferred dtype
                               false_values=["False"],  # no inferred dtype
                               index_col=0,  # first column as index
                               header=0  # first row as header
                               )
            selected_features = read_feature_list(select)
            if selected_features is not None:
                data = data[selected_features]
        else:
            data = pd.read_parquet(data_file, engine="pyarrow", columns=read_feature_list_parquet(select))
            data = data.drop(labels="cluster", errors="ignore", axis=1)  # TODO

        if shuffle_features:
            data = shuffle_columns(data, seed=self.random_state)

        self.features = list(data.columns[:-1])

        if validation:
            X_train, X_test, y_train, y_test = train_test_split(data.drop(self.label_name, axis=1),
                                                                data[[self.label_name]],
                                                                train_size=self.train_frac,
                                                                random_state=self.random_state
                                                                )
            X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                                          y_test,
                                                                          train_size=.5,
                                                                          random_state=self.random_state
                                                                          )
        else:
            X_train, X_test, y_train, y_test = train_test_split(data.drop(self.label_name, axis=1),
                                                                data[[self.label_name]],
                                                                train_size=self.train_frac,
                                                                random_state=self.random_state
                                                                )

        print("Stats (train data):")
        print_stats(X_train, y_train, self.label_name)
        print("Transforming into DMatrices...")
        self.dtrain = xgb.DMatrix(X_train, y_train)
        print()

        if validation:
            print("Stats (validation data):")
            print_stats(X_validation, y_validation, self.label_name)
            print("Transforming into DMatrices...")
            self.dvalidation = xgb.DMatrix(X_validation, y_validation)
            print()
        else:
            self.dvalidation = None

        print("Stats (test data):")
        print_stats(X_test, y_test, self.label_name)
        print("Transforming into DMatrices...")
        self.y_test = y_test
        self.dtest = xgb.DMatrix(X_test)

        print()

        if feature_weights is not None:
            print("Setting weights...")
            assert len(feature_weights) == self.dtrain.num_col()
            self.dtrain.set_info(feature_weights=feature_weights)

        end_t = time.time()
        print(f"Read time {end_t - start_t : .2f}s")

    def set_weights(self, weights=None, equal_weight=False):
        if weights is None:
            weights = self.importance_counts

        fw = []
        for feature in self.features:
            w = weights.get(feature, 0)
            if equal_weight and w > 0:
                w = 1
            fw.append(w)

        self.dtrain.set_info(feature_weights=fw)

    def fit(self, params=None, evals=None):
        if self.dtrain is None:
            raise Exception("Need to load training datasets first!")

        if params is None:
            params = {"verbosity": 1, "device": "cpu", "objective": "binary:hinge", "tree_method": self.method,
                      "colsample_bytree": self.by_tree, "seed": self.random_state,
                      "eta": self.eta, "max_depth": self.max_depth}
            # params["eval_metric"] = "auc"

        if evals is None:
            if self.dvalidation is None:
                evals = [(self.dtrain, "training")]
            else:
                evals = [(self.dtrain, "training"), (self.dvalidation, "validation")]

        self.bst = xgb.train(params=params, dtrain=self.dtrain,
                             num_boost_round=self.num_trees,
                             evals=evals,
                             verbose_eval=5,
                             early_stopping_rounds=self.early_stopping
                             )

        # update number of trees in case of early stopping
        self.num_trees = self.bst.num_boosted_rounds()

        # best values
        self.best_it = self.bst.best_iteration
        self.best_score = self.bst.best_score

        # features importance
        self.importance_counts = self.bst.get_score(importance_type="weight")
        self.importance_gain = self.bst.get_score(importance_type="gain")

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

        assert (true_pos + false_neg) == sum(self.y_test[self.label_name])
        assert (true_neg + false_pos) == len(self.y_test[self.label_name]) - sum(self.y_test[self.label_name])
        assert (true_neg + true_pos + false_neg + false_pos) == len(self.y_test[self.label_name])

        print(f"TN={true_neg}\tFP={false_pos}")
        print(f"FN={false_neg}\tTP={true_pos}")

        # accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        accuracy = accuracy_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        auc = roc_auc_score(self.y_test, self.y_pred)

        print(f"Accuracy = {accuracy * 100 : .3f} %")
        print(f"f1 = {f1 * 100 : .3f} %")
        print(f"ROC_AUC = {auc * 100 : .3f} %")

        num_feat = 100
        importance = sorted(self.importance_counts.items(), key=lambda item: item[1], reverse=True)
        print(f"Top {num_feat}/{len(importance)} features:")
        print(importance[:num_feat])

    def plot_trees(self, tree_set=None, tree_name=None):
        print("Printing trees...")
        if tree_set is None:
            tree_set = range(self.num_trees)

        if tree_name is None:
            tree_name = self.model_name

        for i in tree_set:
            graph: graphviz.Source
            graph = xgb.to_graphviz(self.bst, num_trees=i)
            graph.render(filename=f"{tree_name}-{i}", directory="trees", format="png", cleanup=True)
        print("Done.")

    def write_importance(self, filename):
        with open(filename + ".counts.csv", 'w') as importance_file:
            for item in self.importance_counts.items():
                importance_file.write(f"{item[0]}, {item[1]}\n")

        with open(filename + ".gains.csv", 'w') as importance_file:
            for item in self.importance_gain.items():
                importance_file.write(f"{item[0]}, {item[1]}\n")

    def region_distribution(self):
        narrow_file = "narrow.csv"
        narrow_mock_hk_file = "Hk_mock_narr.csv"
        narrow_mock_br_file = "Br_mock_narr.csv"
        narrow_nvv_hk_file = "Hk_mock_narr.csv"
        narrow_nvv_br_file = "Br_NNV_narr.csv"

        broad_file = "broad.csv"
        broad_mock_hk_file = "Hk_mock_broad.csv"
        broad_mock_br_file = "Br_mock_broad.csv"
        broad_nvv_hk_file = "Hk_mock_broad.csv"
        broad_nvv_br_file = "Br_NNV_broad.csv"

        all_narrow_file = [narrow_file, narrow_nvv_br_file, narrow_nvv_hk_file, narrow_mock_br_file,
                           narrow_mock_hk_file]
        all_broad_file = [broad_file, broad_nvv_br_file, broad_nvv_hk_file, broad_mock_br_file, broad_mock_hk_file]

        def read_and_intersect(file_path, features: set):
            df = pd.read_csv(file_path)
            lst = df.iloc[:, 0].tolist()
            return len(features.intersection(lst))

        s = set(self.importance_counts.keys())
        for b in all_broad_file + all_narrow_file:
            num = read_and_intersect(b, s)
            print(f"{b}: {num} ({num / len(s) * 100} %)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost variant classifier')
    parser.add_argument("--model_name", type=str, default="default-model", help="Model name")
    parser.add_argument("--data", type=str, default="main.csv", help="Input csv/parquet file")
    parser.add_argument("--method", type=str, default="hist", help="Tree method")
    parser.add_argument("--num_trees", type=int, default=100, help="Number of trees")
    parser.add_argument('--validate', default=False, action="store_true")
    parser.add_argument('--shuffle_features', default=False, action="store_true")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth for trees")
    parser.add_argument("--eta", type=float, default=.3, help="Learning rate")
    parser.add_argument("--sample_bytree", type=float, default=2250 / 6072853, help="Sample by tree")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--early_stopping", type=int, default=None, help="Stop after n non-increasing iterations")
    parser.add_argument("--select", type=str, default=None, help="List of feature to select")
    parser.add_argument("--cluster", type=str, default=None, help="List of cluster points for test/train")  # TODO

    args = parser.parse_args()

    clf = XGBoostVariant(model_name=args.model_name, num_trees=args.num_trees, max_depth=args.max_depth, eta=args.eta,
                         sample_bytree=args.sample_bytree, method=args.method, early_stopping=args.early_stopping)
    clf.read_datasets(args.data, validation=args.validate, shuffle_features=args.shuffle_features, select=args.select)

    try:
        os.mkdir(args.model_name)
    except FileExistsError:
        pass
    os.chdir(args.model_name)

    for it in range(args.iterations):
        print(f"\n*** Iteration {it + 1} ***")
        clf.fit()
        clf.predict()
        clf.print_stats()
        clf.write_importance(f"importance-{it}")
        clf.set_weights(equal_weight=True)  # for next iteration

    clf.plot_trees(tree_name="weighted")

# TODO add variable constraints
# TODO add features sampling by node, level
# TODO add data points sampling
# TODO add scale_pos_weight to balance classes
# TODO add gamma (high for conservative algorithm)
