from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import plotly.express as px

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
le_country = LabelEncoder()
le_item = LabelEncoder()
data_cleaned['Country_Encoded'] = le_country.fit_transform(data_cleaned['Area'])
data_cleaned['Item_Encoded'] = le_item.fit_transform(data_cleaned['Item'])

# Define features and target variable
X = data_cleaned[['Country_Encoded', 'Item_Encoded', 'Pesticides', 'Avg_Temp', 'Rainfall']]
y = data_cleaned['Yield']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Bagging Regressor model
bagging_model = BaggingRegressor(n_estimators=100, random_state=1)

# Train the model
bagging_model.fit(X_train, y_train)

# Make predictions
y_pred_bagging = bagging_model.predict(X_test)

# Calculate evaluation metrics
accuracy_train = bagging_model.score(X_train, y_train) * 100
accuracy_test = bagging_model.score(X_test, y_test) * 100
MSE_bagging = mean_squared_error(y_test, y_pred_bagging)
R2_score_bagging = r2_score(y_test, y_pred_bagging)

print(f'Accuracy of Bagging Regressor Model Train is {accuracy_train:.2f}')
print(f'Accuracy of Bagging Regressor Model Test is {accuracy_test:.2f}')
print(f'Mean Squared Error (MSE) of the Bagging Regressor Model: {MSE_bagging:.4f}')
print(f'R^2 Score of the Bagging Regressor Model: {R2_score_bagging:.4f}')

# Create a DataFrame with y_test and y_pred
data = {'y_test': y_test, 'y_pred': y_pred_bagging}
data_df = pd.DataFrame(data)


import matplotlib.pyplot as plt

# Create a scatter plot with trendline using Matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_bagging, color='blue', label='Actual vs. Predicted')
plt.plot(y_test, y_test, color='red', linestyle='--', label='Trendline')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot of Actual vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.show()
