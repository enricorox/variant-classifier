import graphviz
import pandas as pd
import xgboost as xgb
from numpy import ndarray
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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
        self.features = None
        self.importance = None
        self.random_state = 42
        self.model_name = model_name
        self.num_trees = num_trees
        self.label_name = "phenotype"
        self.train_frac = .8

        print(f"Using XGBoost version {xgb.__version__}")

    def read_datasets(self, data_file, validation=False, feature_weights=None):
        print("Loading data...", flush=True)
        data = pd.read_csv(data_file, low_memory=False,
                           true_values=["True"],  # no inferred dtype
                           false_values=["False"],  # no inferred dtype
                           index_col=0,  # first column as index
                           header=0  # first row as header
                           )

        self.features = list(data.columns[:-1])

        if validation:
            X_train, X_test, y_train, y_test = train_test_split(data.drop(self.label_name, axis=1),
                                                                data[[self.label_name]],
                                                                train_size=self.train_frac,
                                                                random_state=self.random_state)
            X_test, X_validation, y_test, y_validation = train_test_split(X_test,
                                                                            y_test,
                                                                            train_size=.5,
                                                                            random_state=self.random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(data.drop(self.label_name, axis=1),
                                                                data[[self.label_name]],
                                                                train_size=self.train_frac,
                                                                random_state=self.random_state)

        print("Stats (train data):")
        print(f"\tData points: {X_train.shape[0]}")
        print(f"\t\tnumber of features: {X_train.shape[1]}")
        print(f"\t\tlabel(0) counts: {(y_train[self.label_name] == 0).sum() / len(y_train[self.label_name]) * 100 : .2f} %")
        print(f"\t\tlabel(1) counts: {(y_train[self.label_name] == 1).sum() / len(y_train[self.label_name]) * 100 : .2f} %")
        print("Transforming into DMatrices...")
        self.dtrain = xgb.DMatrix(X_train, y_train)
        print()

        if validation:
            print("Stats (validation data):")
            print(f"\tData points: {X_validation.shape[0]}")
            print(f"\t\tnumber of features: {X_validation.shape[1]}")
            print(f"\t\tlabel(0) counts: {(y_validation[self.label_name] == 0).sum() / len(y_validation[self.label_name]) * 100 : .2f} %")
            print(f"\t\tlabel(1) counts: {(y_validation[self.label_name] == 1).sum() / len(y_validation[self.label_name]) * 100 : .2f} %")
            print("Transforming into DMatrices...")
            self.dvalidation = xgb.DMatrix(X_validation, y_validation)
            print()
        else:
            self.dvalidation = None

        print("Stats (test data):")
        print(f"\tData points: {X_test.shape[0]}")
        print(f"\t\tnumber of features: {X_train.shape[1]}")
        print(f"\t\tlabel(0) counts: {(y_test[self.label_name] == 0).sum() / len(y_test[self.label_name]) * 100 : .2f} %")
        print(f"\t\tlabel(1) counts: {(y_test[self.label_name] == 1).sum() / len(y_test[self.label_name]) * 100 : .2f} %")
        print("Transforming into DMatrices...")
        self.y_test = y_test
        self.dtest = xgb.DMatrix(X_test)

        print()

        if feature_weights is not None:
            print("Setting weights...")
            assert len(feature_weights) == self.dtrain.num_col()
            self.dtrain.set_info(feature_weights=feature_weights)

    def set_weights(self, weights=None, equal_weight=False):
        if weights is None:
            weights = self.importance

        fw = []
        for feature in self.features:
            w = weights.get(feature, 1)
            if equal_weight and w > 1:
                w = 2
            fw.append(w)

        self.dtrain.set_info(feature_weights=fw)

    def fit(self, params=None, evals=None):
        if self.dtrain is None:
            raise Exception("Need to load training datasets first!")

        if params is None:
            params = {"verbosity": 1, "device": "cpu", "objective": "binary:hinge", "tree_method": "hist",
                      "colsample_bytree": .75, "seed": self.random_state,
                      "eta": .3, "max_depth": 6}
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
                             early_stopping_rounds=50
                             )

        # update number of trees in case of early stopping
        self.num_trees = self.bst.num_boosted_rounds()
        self.best_it = self.bst.best_iteration
        self.best_score = self.bst.best_score

        # features importance
        self.importance = self.bst.get_score(importance_type="weight")

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

        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        print(f"Accuracy = {accuracy * 100 : .2f} %")

        num_feat = 100
        importance = sorted(self.importance.items(), key=lambda item: item[1], reverse=True)
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


if __name__ == "__main__":
    dataset_folder = "./"
    data_file = dataset_folder + "main.csv"

    clf = XGBoostVariant(num_trees=100)
    clf.read_datasets(data_file, validation=False)

    for it in range(10):
        print(f"\n*** Iteration {it + 1} ***")
        clf.fit()
        clf.predict()
        clf.print_stats()
        clf.set_weights(equal_weight=True)  # for next iteration

    clf.plot_trees(tree_name="weighted")

# TODO add max_depth
# TODO add variable constraints
# TODO add features sampling by *
# TODO add data points sampling
# TODO add scale_pos_weight to balance classes
# TODO add eta (learning rate)
# TODO add gamma (high for conservative algorithm)
