from xgboost import XGBRegressor
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


# Initialize the XGBoost model
xgb_model = XGBRegressor(random_state=1)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Calculate evaluation metrics
accuracy_train = xgb_model.score(X_train, y_train) * 100
accuracy_test = xgb_model.score(X_test, y_test) * 100
MSE_xgb = mean_squared_error(y_test, y_pred_xgb)
R2_score_xgb = r2_score(y_test, y_pred_xgb)

print(f'Accuracy of XGBoost Model Train is {accuracy_train:.2f}')
print(f'Accuracy of XGBoost Model Test is {accuracy_test:.2f}')
print(f'Mean Squared Error (MSE) of the XGBoost Model: {MSE_xgb:.4f}')
print(f'R^2 Score of the XGBoost Model: {R2_score_xgb:.4f}')

# Create a DataFrame with y_test and y_pred
data = {'y_test': y_test, 'y_pred': y_pred_xgb}
data_df = pd.DataFrame(data)


import matplotlib.pyplot as plt
# Extract data for plotting
actual_values = data_df['y_test']
predicted_values = data_df['y_pred']

# Create the scatter plot
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.scatter(actual_values, predicted_values, alpha=0.7, label='Actual vs Predicted')

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (XGBoost)')

# Add trendline (optional)
m, b = np.polyfit(actual_values, predicted_values, 1)  # Linear regression
plt.plot(actual_values, m * actual_values + b, color='red', label='Trendline')
plt.legend()

# Grid lines (optional)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', axis='y')

# Display the plot within Python
plt.show()
