from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import pandas as pd
import numpy as np
import plotly.graph_objs as go

# Load the dataset
dataset_path = r"C:\Rpro\labtest\yield_df.csv"
data = pd.read_csv(dataset_path)

# Rename columns for clarity
data_renamed = data.rename(columns={
    "hg/ha_yield": "Yield",
    "average_rain_fall_mm_per_year": "Rainfall",
    "pesticides_tonnes": "Pesticides",
    "avg_temp": "Avg_Temp"
})

# Drop the index column if it's not needed
data_cleaned = data_renamed.drop(columns=["Unnamed: 0"])

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder

le_country = LabelEncoder()
le_item = LabelEncoder()
data_cleaned['Country_Encoded'] = le_country.fit_transform(data_cleaned['Area'])
data_cleaned['Item_Encoded'] = le_item.fit_transform(data_cleaned['Item'])

# Define features and target variable
X = data_cleaned[['Country_Encoded', 'Item_Encoded', 'Pesticides', 'Avg_Temp', 'Rainfall']]
y = data_cleaned['Yield']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# K-Fold Cross Validation
results = []
fold_df = pd.DataFrame()

# Loop through the list of machine learning models
models = [
    ('DecisionTree', DecisionTreeRegressor(random_state=1)),
    ('RandomForest', RandomForestRegressor(random_state=1)),
    ('BaggingRegressor', BaggingRegressor(n_estimators=100, random_state=1)),
    ('KNN', KNeighborsRegressor()),
    ('XGBoost', XGBRegressor(random_state=1)),
]

for name, model in models:
    # Train Model
    model.fit(X_train, y_train)
    # Make Predictions
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    MSE = mean_squared_error(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    R2_score = r2_score(y_test, y_pred)
    # Add all metrics of model to a list
    results.append((name, accuracy, MSE, MAE, MAPE, R2_score))

    print(name)
    kf = KFold(n_splits=10, shuffle=True)
    scores = cross_val_score(model, X, y, cv=kf)

    # Print out the CV Scores for each fold
    for fold, score in enumerate(scores):
        print(f'Fold {fold + 1}: {score}')
        temp_df = pd.DataFrame({'Name': name, 'Fold': [fold + 1], 'Score': [score]})
        dfs = [fold_df, temp_df]
        fold_df = pd.concat(dfs, ignore_index=True)

    # Print out the Mean CV scores for each model
    mean_score = np.mean(scores)
    print(f'Mean Score: {mean_score}')
    print('=' * 30)


import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd

# Define your models and fold_df here

for i in range(len(models)):
    plot_fold_df = fold_df[fold_df['Name'] == models[i][0]]

    plt.figure()

    plt.plot(plot_fold_df['Fold'], plot_fold_df['Score'], marker='o')
    plt.title(models[i][0])
    plt.xlabel('Fold')
    plt.ylabel('Score')

    plt.show()


# Dataframe consisting of metrics of all the models
result_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'MSE', 'MAE', 'MAPE', 'R2_score'])

print(result_df)
