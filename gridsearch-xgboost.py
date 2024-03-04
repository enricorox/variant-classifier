import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split

data_file = "main-012.csv"
data = pd.read_csv(data_file, low_memory=False,
                   index_col=0,  # first column as index
                   header=0  # first row as header
                   )
y = data["phenotype"]
data.drop(columns=["phenotype", "cluster"])
X = data.sample(frac=1, axis=1, random_state=42)

# Dividi il dataset in set di addestramento e set di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definisci la griglia degli iperparametri da testare
param_grid = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [1, 2, 3],
    # 'subsample': [0.7, 0.8, 0.9, 1],
    'colsample_bytree': [0.7, 0.8, 0.9, 1],
    'grow_policy': ['depthwise', 'lossguide']
}

# Inizializza il regressore XGBoost
xgb_model = xgb.XGBRegressor()

# Crea l'oggetto GridSearchCV
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=3)

# Esegui la ricerca a griglia sull'insieme di addestramento
grid_search.fit(X_train, y_train)

# Stampare i migliori iperparametri trovati
print(f"Migliori iperparametri trovati: {grid_search.best_params_}")
print(f"Migliore validation score: {grid_search.best_score_}")

# Valutare le prestazioni del modello migliore sui dati di test
best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Punteggio sul set di test:", test_score)
