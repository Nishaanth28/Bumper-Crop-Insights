
import pandas as pd
from tabulate import tabulate

# Load the dataset
dataset_path = r"C:\Rpro\labtest\yield_df.csv"
df = pd.read_csv(dataset_path)


results = [
    {'Model': 'Bagging Regressor', 'Accuracy_train': 99.61, 'Accuracy_test': 97.28, 'MSE': 197611608.7566, 'R2_score': 0.9728},
    {'Model': 'Decision Tree', 'Accuracy_train': 99.96, 'Accuracy_test': 95.65, 'MSE': 315321644.72, 'R2_score': 0.96},
    {'Model': 'XGBoost', 'Accuracy_train': 97.74, 'Accuracy_test': 96.60, 'MSE': 246811225.8903, 'R2_score': 0.9660},
    {'Model':'RandomForest','Accuracy_train': 99.61, 'Accuracy_test': 97.28, 'MSE': 197241168.6922, 'R2_score': 0.9728},
    {'Model': 'KNN', 'Accuracy_train': 0.49, 'Accuracy_test': 0.38, 'MSE': 4521871033.8061, 'R2_score': 0.3766}
]

print(results)

# Extract keys from the first dictionary to ensure consistent order
headers = results[0].keys()

# Extract values from all dictionaries
data = [[row[key] for key in headers] for row in results]

# Print the results in a tabular format
print(tabulate(data, headers=headers, tablefmt="grid"))

