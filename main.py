
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

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from tkinter import messagebox

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
# Define feature names
feature_names = X.columns.tolist()
# Load the Decision Tree model
decision_tree_model = DecisionTreeRegressor(random_state=1)
decision_tree_model.fit(X_train, y_train)

# Load label encoders
le_country.fit(data_cleaned['Area'])
le_item.fit(data_cleaned['Item'])

# Function to predict crop yield
def predict_yield(country, item, pesticides, avg_temp, rainfall):
    # Preprocess user input
    if country not in data_cleaned['Area'].unique():
        st.warning("Country not found in dataset. Please enter a valid country name.")
        return
    # Preprocess user input
    country_encoded = le_country.transform([country])[0]
    item_encoded = le_item.transform([item])[0]

    # Make prediction using the Decision Tree model
    features = [[country_encoded, item_encoded, pesticides, avg_temp, rainfall]]
    predicted_yield = decision_tree_model.predict(features)

    # Display prediction
    st.info(f"Predicted Yield: {predicted_yield[0]:.2f} hg/ha")


# Load and display image
image_path = r"C:\Rpro\labtest\SDG2.png"

st.image(image_path, caption='Zero Hunger', width= 64)
# Create Streamlit app
def main():
    st.title("Bumper Crop Insights Yield Prediction")

    # Create input fields for user input
    country = st.selectbox("Select Country", data_cleaned['Area'].unique()) # Create dropdown list for country selection
    items_for_selected_country = data_cleaned[data_cleaned['Area'] == country]['Item'].unique()
    item = st.selectbox("Select Item", items_for_selected_country)  # Create dropdown list for item selection
    pesticides = st.number_input("Pesticides (tonnes):")
    avg_temp = st.number_input("Average Temperature (°C):")
    rainfall = st.number_input("Rainfall (mm/year):")

    # Button to predict yield
    if st.button("Predict Yield"):
        predict_yield(country, item, pesticides, avg_temp, rainfall)

if __name__ == "__main__":
    main()
