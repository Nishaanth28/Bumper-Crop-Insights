from sklearn.tree import DecisionTreeRegressor
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



import matplotlib.pyplot as plt

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

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

# Load the Decision Tree model
decision_tree_model = DecisionTreeRegressor(random_state=1)
decision_tree_model.fit(X_train, y_train)

# Load label encoders
le_country.fit(data_cleaned['Area'])
le_item.fit(data_cleaned['Item'])


# Function to predict crop yield
def predict_yield(country, item, pesticides, avg_temp, rainfall):
    # Preprocess user input
    country_encoded = le_country.transform([country])[0]
    item_encoded = le_item.transform([item])[0]

    # Make prediction using the Decision Tree model
    features = [[country_encoded, item_encoded, pesticides, avg_temp, rainfall]]
    predicted_yield = decision_tree_model.predict(features)

    # Display prediction
    messagebox.showinfo("Prediction", f"Predicted Yield: {predicted_yield[0]:.2f} hg/ha")


# Create main application window
root = tk.Tk()
root.title("Crop Yield Prediction")

# Create labels and entry fields for user input
country_label = ttk.Label(root, text="Country:")
country_label.grid(row=0, column=0, padx=10, pady=5)
country_entry = ttk.Entry(root)
country_entry.grid(row=0, column=1, padx=10, pady=5)

item_label = ttk.Label(root, text="Item:")
item_label.grid(row=1, column=0, padx=10, pady=5)
item_entry = ttk.Entry(root)
item_entry.grid(row=1, column=1, padx=10, pady=5)

pesticides_label = ttk.Label(root, text="Pesticides (tonnes):")
pesticides_label.grid(row=2, column=0, padx=10, pady=5)
pesticides_entry = ttk.Entry(root)
pesticides_entry.grid(row=2, column=1, padx=10, pady=5)

temp_label = ttk.Label(root, text="Average Temperature (°C):")
temp_label.grid(row=3, column=0, padx=10, pady=5)
temp_entry = ttk.Entry(root)
temp_entry.grid(row=3, column=1, padx=10, pady=5)

rainfall_label = ttk.Label(root, text="Rainfall (mm/year):")
rainfall_label.grid(row=4, column=0, padx=10, pady=5)
rainfall_entry = ttk.Entry(root)
rainfall_entry.grid(row=4, column=1, padx=10, pady=5)

# Button to predict yield
predict_button = ttk.Button(root, text="Predict Yield", command=lambda: predict_yield(country_entry.get(),
                                                                                      item_entry.get(),
                                                                                      float(pesticides_entry.get()),
                                                                                      float(temp_entry.get()),
                                                                                      float(rainfall_entry.get())))
predict_button.grid(row=5, columnspan=2, padx=10, pady=5)

root.mainloop()

