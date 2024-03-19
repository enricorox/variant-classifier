import argparse
import os
import time

import graphviz
import numpy as np
import pandas as pd
import xgboost as xgb
from numpy import ndarray
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import Booster


def read_feature_list(selection_file):
    features = pd.read_csv(selection_file, header=None)
    print(f"Read {len(features)} features to select")
    return features.iloc[:, 0].tolist()


def print_stats(X, y, label):
    print(f"\tData points: {X.shape[0]}")
    print(f"\t\tnumber of features: {X.shape[1]}")
    print(f"\t\tlabel(0) counts: {(y[label] == 0).sum() / len(y[label]) * 100 : .2f} %")
    print(f"\t\tlabel(1) counts: {(y[label] == 1).sum() / len(y[label]) * 100 : .2f} %")


def read_cluster_file(cluster_file):
    points = pd.read_csv(cluster_file, header=0, index_col=0)
    clusters = [points.index[points["cluster"] == 0],
                points.index[points["cluster"] == 1]]
    print(f"clusters: {len(clusters[0])}, {len(clusters[1])}")
    return clusters


def group_by_chromosome(weigths, gains):
    keys = weigths.keys()
    counts_group = [0] * 24
    weigths_group = [0] * 24
    gains_group = [0] * 24
    for key in keys:
        chromosome = int(key[13:15]) - 1  # e.g. CAJNNU010000001.1:4119343
        counts_group[chromosome] += 1
        weigths_group[chromosome] += weigths[key]
        gains_group[chromosome] += gains[key]

    return counts_group, weigths_group, gains_group


def group_by_region(weigths, gains):
    # List of CSV file paths
    regions_folder = "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/exclude-chr3/regions/"
    regions_files = ["Br_mock_broad.csv", "Br_NNV_broad.csv", "Hk_mock_broad.csv", "Hk_NNV_broad.csv", "broad.csv",
                     "Br_mock_narr.csv", "Br_NNV_narr.csv", "Hk_mock_narr.csv", "Hk_NNV_narr.csv", "narrow.csv"]
    data_ensemble_file = "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/exclude-chr3/data_ensemble.csv"

    # Create a list to store lists of strings from each CSV file
    total_gain = gains.loc[:, 1].sum()

    total_counts = weigths.loc[:, 1].sum()
    features = set(weigths.index)

    # Read from each CSV file and append the list to the main list
    print("Peaks\tCounts\tWeights\tGains")
    for file in regions_files:
        current_list = pd.read_csv(regions_folder + file).iloc[:, 0].tolist()
        common_strings = features.intersection(current_list)

        grouped_gains = gains.loc[list(common_strings), 1]
        grouped_counts = weigths.loc[list(common_strings), 1]
        print(f"{file}\t"
              f"{len(common_strings) / len(features) * 100 : .2f} %\t"
              f"{grouped_gains.sum() / total_gain * 100 : .2f} %\t"
              f"{grouped_counts.sum() / total_counts * 100 : .2f}%")
        if file == "broad.csv":
            print()

def data_ensemble(features, gains):
    data_ensemble_file = "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/exclude-chr3/data_ensemble.csv"
    features = set(features)

    data_ensemble = pd.read_csv(data_ensemble_file, index_col=0, header=0)
    ensemble_features = list(features.intersection(data_ensemble.index.values))
    data_ensemble = data_ensemble.loc[ensemble_features, :]
    data_ensemble["gain"] = gains.loc[ensemble_features, :]
    data_ensemble.sort_values(by="gain")
    data_ensemble.to_csv("ensemble.csv")

    print(f"#features in the ensemble: {len(ensemble_features)}")

    if len(ensemble_features) > 0:
        info_funct = data_ensemble["funct"].value_counts()
        info_n_tissue = data_ensemble["n_tissue"].value_counts()

        print(info_funct)
        print(info_n_tissue)
    else:
        print("No features in the data ensemble!!!")


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

    target: str
    features = None
    importance_weights = None
    importance_gains = None

    def __init__(self, model_name, num_trees, max_depth, eta, early_stopping,
                 sample_bytree, method, objective,
                 data_file, target_file, validation=False, do_shuffle_features=False,
                 selected_features_file=None, cluster_file=None, invc=False
                 ):
        self.auc = None
        self.f1 = None
        self.accuracy = None
        self.num_features = None
        self.max_depth = max_depth
        self.eta = eta
        self.early_stopping = early_stopping
        self.by_tree = sample_bytree
        self.random_state = 42
        self.model_name = model_name
        self.num_trees = num_trees
        self.train_frac = .8
        self.method = method
        self.objective = objective  # binary:logistic or reg:logistic

        if early_stopping is None:
            self.early_stopping = self.num_trees

        self.data_file = data_file
        self.target_file = target_file
        self.validation = validation
        self.do_shuffle_features = do_shuffle_features
        self.selected_features_file = selected_features_file
        self.cluster_file = cluster_file
        self.invc = invc

        print(f"Using XGBoost version {xgb.__version__}")

    def read_datasets(self, ):
        data_file = self.data_file
        target_file = self.target_file
        validation = self.validation
        do_shuffle_features = self.do_shuffle_features
        selected_features_file = self.selected_features_file
        cluster_file = self.cluster_file
        invc = self.invc

        start_t = time.time()
        print("Loading features...", flush=True)
        data = pd.read_csv(data_file, low_memory=False,
                           index_col=0,  # first column as index
                           header=0  # first row as header
                           ).astype(np.int8)
        print("Done.", flush=True)

        if selected_features_file is not None:
            print("Selecting features...", flush=True)
            selected_features = read_feature_list(selected_features_file)
            data = data[selected_features]
            print("Done.", flush=True)

        if do_shuffle_features:
            print("Shuffling features...", flush=True)
            start_shuffle_t = time.time()
            data = data.sample(frac=1, axis=1, random_state=self.random_state)
            stop_shuffle_t = time.time()
            print(f"Done in {stop_shuffle_t - start_shuffle_t} s", flush=True)

        self.features = list(data.columns)

        print("Reading targets...", flush=True)
        labels = pd.read_csv(target_file, header=0, index_col=0)
        self.target = labels.columns[0]
        print(f"Target is {self.target}")
        print("Done.", flush=True)

        print("Splitting the datasets...", flush=True)
        if cluster_file is None:
            if validation:
                X_train, X_test, y_train, y_test = train_test_split(data,
                                                                    labels,
                                                                    train_size=self.train_frac,
                                                                    random_state=self.random_state
                                                                    )
                X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                                              y_test,
                                                                              train_size=.5,
                                                                              random_state=self.random_state
                                                                              )
            else:
                X_train, X_test, y_train, y_test = train_test_split(data,
                                                                    labels,
                                                                    train_size=self.train_frac,
                                                                    random_state=self.random_state
                                                                    )
        else:
            clusters = read_cluster_file(cluster_file)
            if invc:
                cluster_train = clusters[1]
                cluster_test = clusters[0]
            else:
                cluster_train = clusters[0]
                cluster_test = clusters[1]

            X_train = data.iloc[cluster_train, :]
            y_train = labels.iloc[cluster_train]  # Series
            y_train = pd.DataFrame(y_train)

            X_test = data.iloc[cluster_test, :]
            y_test = labels.iloc[cluster_test]  # Series
            y_test = pd.DataFrame(y_test)
        print("Done.\n", flush=True)

        print("Stats (train data):")
        print_stats(X_train, y_train, self.target)
        print("Transforming X_train and y_train into DMatrices...")
        self.dtrain = xgb.DMatrix(X_train, y_train)
        print()

        if validation:
            print("Stats (validation data):")
            print_stats(X_validation, y_validation, self.target)
            print("Transforming X_validation and y_validation into DMatrices...")
            self.dvalidation = xgb.DMatrix(X_validation, y_validation)
            print()
        else:
            self.dvalidation = None

        print("Stats (test data):")
        print_stats(X_test, y_test, self.target)
        print("Transforming X_test into DMatrices...")
        self.y_test = y_test
        self.dtest = xgb.DMatrix(X_test)

        print()

        end_t = time.time()
        print(f"Read time {end_t - start_t : .2f}s")

    def set_weights(self, weights=None, equal_weight=False):
        # feature weights TODO fix random order with dictionary ow weights!!!
        """
        if feature_weights is not None:
            print("Setting weights...")
            assert len(feature_weights) == self.dtrain.num_col()
            self.dtrain.set_info(feature_weights=feature_weights)
        """
        if weights is None:
            weights = self.importance_weights

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
            params = {"verbosity": 1, "device": "cpu", "objective": self.objective, "tree_method": self.method,
                      "seed": self.random_state,
                      "eta": self.eta, "max_depth": self.max_depth}
            if self.by_tree < 1:
                params["colsample_bytree"] = self.by_tree
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
        self.importance_weights = self.bst.get_score(importance_type="weight")
        self.importance_gains = self.bst.get_score(importance_type="gain")

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

        assert (true_pos + false_neg) == sum(self.y_test[self.target])
        assert (true_neg + false_pos) == len(self.y_test[self.target]) - sum(self.y_test[self.target])
        assert (true_neg + true_pos + false_neg + false_pos) == len(self.y_test[self.target])

        print(f"TN={true_neg}\tFP={false_pos}")
        print(f"FN={false_neg}\tTP={true_pos}")

        # accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.f1 = f1_score(self.y_test, self.y_pred)
        self.auc = roc_auc_score(self.y_test, self.y_pred)

        print(f"Accuracy = {self.accuracy * 100 : .3f} %")
        print(f"f1 = {self.f1 * 100 : .3f} %")
        print(f"ROC_AUC = {self.auc * 100 : .3f} %")

        print_num_feat = 100
        importance = sorted(self.importance_weights.items(), key=lambda item: item[1], reverse=True)
        self.num_features = len(importance)
        print(f"Top {print_num_feat}/{self.num_features} features:")
        print(importance[:print_num_feat])

    def write_stats(self, stats_file="stats.csv"):
        with open(stats_file, 'w') as stats:
            stats.write(f"method name,{self.model_name}\n")
            stats.write(f"algorithm,{self.method}\n")
            stats.write(f"training set,{self.train_frac * 100}%\n")
            stats.write(f"validation set,{self.validation}\n")
            stats.write(f"feature shuffle,{self.do_shuffle_features}\n")
            stats.write(f"feature sampling,{self.by_tree}\n")
            stats.write(f"feature peaks,{self.selected_features_file}\n")
            stats.write(f"early stopping,{self.early_stopping}\n")
            stats.write(f"#trees,{self.num_trees}\n")
            stats.write(f"eta,{self.eta}\n")
            stats.write(f"max depth,{self.max_depth}\n")

            stats.write("\n")

            stats.write(f"#features,{self.num_features}\n")
            stats.write(f"f1,{self.f1}\n")
            stats.write(f"ROC AUC,{self.auc}\n")
            stats.write(f"best iteration,{self.best_it}\n")
            stats.write(f"#tree created,{self.bst.num_boosted_rounds()}\n")

            stats.write("\n")

            stats.write("CHR,Count,Weight,Gain\n")
            counts, weights, gains = group_by_chromosome(self.importance_weights, self.importance_gains)
            for i in range(24):
                stats.write(f"{i + 1},{counts[i]},{weights[i]},{gains[i]}\n")

            stats.write("\n")

            # TODO complete for sheet!

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
            for item in self.importance_weights.items():
                importance_file.write(f"{item[0]}, {item[1]}\n")

        with open(filename + ".gains.csv", 'w') as importance_file:
            for item in self.importance_gains.items():
                importance_file.write(f"{item[0]}, {item[1]}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='XGBoost variant classifier')
    parser.add_argument("--model_name", type=str, default="default-model", help="Model name")
    parser.add_argument("--dataset", type=str, default="features.csv",
                        help="Features csv file")
    parser.add_argument("--target", type=str, default="mortality.csv", help="Target csv file")
    parser.add_argument("--select", type=str, default=None, help="List of feature to select")
    parser.add_argument("--cluster", type=str, default=None, help="List of cluster points for test/train")
    parser.add_argument('--invc', default=False, action="store_true",
                        help="use cluster 2 for training and cluster 1 for testing")

    parser.add_argument("--method", type=str, default="exact", help="Tree method")
    parser.add_argument("--num_trees", type=int, default=100, help="Number of trees")
    parser.add_argument('--validate', default=False, action="store_true")
    parser.add_argument('--shuffle_features', default=True, action="store_true")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth for trees")
    parser.add_argument("--eta", type=float, default=.2, help="Learning rate")
    parser.add_argument("--sample_bytree", type=float, default=1, help="Sample by tree")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    parser.add_argument("--early_stopping", type=int, default=None, help="Stop after n non-increasing iterations")
    parser.add_argument('--objective', default="binary:hinge", help="binary:hinge or binary:logistic or...")

    args = parser.parse_args()

    print(args)

    clf = XGBoostVariant(model_name=args.model_name, num_trees=args.num_trees,
                         max_depth=args.max_depth, eta=args.eta,
                         sample_bytree=args.sample_bytree, method=args.method, early_stopping=args.early_stopping,
                         objective=args.objective,
                         data_file=args.dataset, target_file=args.target, validation=args.validate,
                         do_shuffle_features=args.shuffle_features,
                         selected_features_file=args.select, cluster_file=args.cluster, invc=args.invc
                         )
    clf.read_datasets()

    try:
        os.mkdir(args.model_name)
    except FileExistsError:
        print(f"Warning: overwriting existing files in {args.model_name}")
    os.chdir(args.model_name)

    for it in range(args.iterations):
        print(f"\n*** Iteration {it + 1} ***")
        clf.fit()
        clf.predict()
        clf.print_stats()
        clf.write_importance(f"importance-{it}")
        clf.set_weights(equal_weight=True)  # for next iteration

    clf.plot_trees(tree_name="weighted")
    clf.write_stats()

# TODO add variable constraints
# TODO add features sampling by node, level
# TODO add data points sampling
# TODO add scale_pos_weight to balance classes
# TODO add gamma (high for conservative algorithm)
# TODO add random forest (parallel trees)
