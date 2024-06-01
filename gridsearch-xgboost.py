import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split


def read_feature_list(selection_file):
    features = pd.read_csv(selection_file, header=None)
    print(f"Read {len(features)} features to select")
    return features.iloc[:, 0].tolist()


data_file = "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features.csv"
label_file = "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/mortality.csv"
train_set_file = "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/cluster-0.csv"
selected_features_file = "/nfsd/bcb/bcbg/rossigno/PNRR/variant-classifier/datasets/include-chr3/features-sets/Hk_NNV_broad.csv"

print("Reading data...", flush=True)

data = pd.read_csv(data_file, low_memory=False,
                   index_col=0,  # first column as index
                   header=0  # first row as header
                   )
y = pd.read_csv(label_file, low_memory=False,
                index_col=0,  # first column as index
                header=0  # first row as header
                )

X = data.sample(frac=1, axis=1, random_state=42)

print(f"Reading training set IDs...", flush=True)
train_cluster = pd.read_csv(train_set_file, header=0)["id"].values.tolist()

print("Selecting features...", flush=True)
selected_features = read_feature_list(selected_features_file)
data = data[selected_features]
print("Done.", flush=True)

X_train = X.loc[train_cluster]
y_train = y.loc[train_cluster]  # Series
y_train = pd.DataFrame(y_train)

X_test = data.drop(train_cluster)
y_test = y.drop(train_cluster)  # Series
y_test = pd.DataFrame(y_test)

param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [4, 5, 6],
    'min_child_weight': [1, 2, 3],
    # 'subsample': [0.7, 0.8, 0.9, 1],
    # 'colsample_bytree': [0.7, 0.8, 1],
    'grow_policy': ['lossguide', 'depthwise']
}

xgb_model = xgb.XGBClassifier()

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=1, pre_dispatch=1,
                           # scoring='neg_mean_squared_error',
                           error_score=np.Inf, verbose=3)

print("\n*** Starting grid search... ***\n", flush=True)
grid_search.fit(X_train, y_train)

# grid_search.get_params()
print(f"Migliori iperparametri trovati:\n{grid_search.best_params_}")
print(f"Migliore validation score: {grid_search.best_score_}")

# Valutare le prestazioni del modello migliore sui dati di test
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Punteggio sul set di test:", test_score)
