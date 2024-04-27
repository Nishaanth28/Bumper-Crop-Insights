from sklearn.neighbors import KNeighborsRegressor
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


# Initialize the KNN model
knn_model = KNeighborsRegressor(n_neighbors=10)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn_model.predict(X_test)

# Calculate evaluation metrics
accuracy_train = knn_model.score(X_train, y_train)
accuracy_test = knn_model.score(X_test, y_test)
MSE_knn = mean_squared_error(y_test, y_pred_knn)
R2_score_knn = r2_score(y_test, y_pred_knn)

print(f'Accuracy of KNN Model Train is {accuracy_train:.2f}')
print(f'Accuracy of KNN Model Test is {accuracy_test:.2f}')
print(f'Mean Squared Error (MSE) of the KNN Model: {MSE_knn:.4f}')
print(f'R^2 Score of the KNN Model: {R2_score_knn:.4f}')

# Create a DataFrame with y_test and y_pred
data = {'y_test': y_test, 'y_pred': y_pred_knn}
data_df = pd.DataFrame(data)

# Create scatter plot using Plotly Express
fig = px.scatter(data_df, x='y_test', y='y_pred',
                 labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                 trendline='ols', trendline_color_override='red',
                 template='plotly_dark')
fig.show()

import matplotlib.pyplot as plt

# Create a scatter plot with trendline using matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_knn, color='blue', label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Trendline')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (KNN Model)')
plt.legend()
plt.grid(True)
plt.show()

