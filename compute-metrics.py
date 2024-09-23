import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, accuracy_score, \
    mean_squared_error

# Load the CSV file
df = pd.read_csv('predictions.csv')

true_labels = df['mortality']
predictions = df['pred']

# Compute the metrics
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)
roc_auc = roc_auc_score(true_labels, predictions)
matthews = matthews_corrcoef(true_labels, predictions)
accuracy = accuracy_score(true_labels, predictions)
rmse = mean_squared_error(true_labels, predictions, squared=False)

# Print the metrics
print("precision, recall, ACC, f1, ROC AUC, MCC, RMSE")
print(f"{precision}, {recall}, {accuracy}, {f1}, {roc_auc}, {matthews}, {rmse}")

print()

print(f"precision, {precision}")
print(f"recall, {recall}")
print(f"ACC, {accuracy}")
print(f"f1, {f1}")
print(f"ROC AUC, {roc_auc}")
print(f"MCC, {matthews}")
print(f"RMSE, {rmse}")
