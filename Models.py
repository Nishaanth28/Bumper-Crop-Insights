from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor

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
##################################################################################################################################
##################################################################################################################################
# Initialize the Random Forest model
rf_model = RandomForestRegressor(random_state=1)
# Train the model
rf_model.fit(X_train, y_train)
# Make predictions
y_pred_rf = rf_model.predict(X_test)
# Calculate evaluation metrics
accuracy_train = rf_model.score(X_train, y_train) * 100
accuracy_test = rf_model.score(X_test, y_test) * 100
MSE_rf = mean_squared_error(y_test, y_pred_rf)
R2_score_rf = r2_score(y_test, y_pred_rf)
print(f'Accuracy of Random Forest Model Train is {accuracy_train:.2f}')
print(f'Accuracy of Random Forest Model Test is {accuracy_test:.2f}')
print(f'Mean Squared Error (MSE) of the Random Forest Model: {MSE_rf:.4f}')
print(f'R^2 Score of the Random Forest Model: {R2_score_rf:.4f}')
# Create a DataFrame with y_test and y_pred
data = {'y_test': y_test, 'y_pred': y_pred_rf}
data_df = pd.DataFrame(data)
# Extract data for plotting
actual_values = data_df['y_test']
predicted_values = data_df['y_pred']


# Create the scatter plot
plt.figure(figsize=(8, 6))  # Adjust figure size as needed
plt.scatter(actual_values, predicted_values, alpha=0.7, label='Data Points')
# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Random Forest)')  #
# Add trendline (optional)
m, b = np.polyfit(actual_values, predicted_values, 1)  # Linear regression
plt.plot(actual_values, m * actual_values + b, color='red', label='Trendline')
plt.legend()
# Grid lines (optional)
plt.grid(True, linestyle='--', linewidth=0.5, color='gray', which='both', axis='y')
# Display the plot within Python
plt.show()

######################################################################################################
######################################################################################################

# Define the Decision Tree model
decision_tree_model = DecisionTreeRegressor(random_state=1)
# Train the Decision Tree model
decision_tree_model.fit(X_train, y_train)
# Make predictions
y_pred = decision_tree_model.predict(X_test)

# Calculate evaluation metrics
accuracy_train = decision_tree_model.score(X_train, y_train) * 100
accuracy_test = decision_tree_model.score(X_test, y_test) * 100
mse = mean_squared_error(y_test, y_pred)
r2_score = r2_score(y_test, y_pred)
# Print evaluation metrics
print(f"Accuracy of Decision Tree Model Train: {accuracy_train:.2f}")
print(f"Accuracy of Decision Tree Model Test: {accuracy_test:.2f}")
print(f"Mean Squared Error of Decision Tree Model Test: {mse:.2f}")
print(f"R^2 Score of Decision Tree Model Test: {r2_score:.2f}")
# Create a DataFrame for visualization
data = {'y_test': y_test, 'y_pred': y_pred}
data_df = pd.DataFrame(data)

# Create a scatter plot with trendline using matplotlib
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red', label='Trendline')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values (Decision tree)')
plt.legend()
plt.grid(True)
plt.show()

###########################################################################################################################
###########################################################################################################################

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

#####################################################################################################
#####################################################################################################

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

##############################################################################################################################
##############################################################################################################################

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
