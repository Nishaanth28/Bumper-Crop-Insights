import os.path

import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
dataset_path = os.path.join('data/yield_df.csv')
data = pd.read_csv(dataset_path)

# Preprocess the dataset
data = data.rename(columns={
    "hg/ha_yield": "Yield",
    "average_rain_fall_mm_per_year": "Rainfall",
    "pesticides_tonnes": "Pesticides",
    "avg_temp": "Avg_Temp"
})

# Encode categorical variables
le_country = LabelEncoder()
le_item = LabelEncoder()
data['Country_Encoded'] = le_country.fit_transform(data['Area'])
data['Item_Encoded'] = le_item.fit_transform(data['Item'])

# Define features and target variable
X = data[['Country_Encoded', 'Item_Encoded', 'Pesticides', 'Avg_Temp', 'Rainfall']]
y = data['Yield']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
decision_tree_model = DecisionTreeRegressor(random_state=1)
decision_tree_model.fit(X_train, y_train)


# Ensure that X_test has the same column names as X_train
X_test.columns = ['Country_Encoded', 'Item_Encoded', 'Pesticides', 'Avg_Temp', 'Rainfall']

# Make predictions with explicit feature names
y_pred = decision_tree_model.predict(X_test)
# Function to predict crop yield
def predict_yield(country, item, pesticides, avg_temp, rainfall):
    # Preprocess user input
    country_encoded = le_country.transform([country])[0]
    item_encoded = le_item.transform([item])[0]

    # Make prediction using the Decision Tree model
    features = [[country_encoded, item_encoded, pesticides, avg_temp, rainfall]]
    predicted_yield = decision_tree_model.predict(features)

    # Display prediction
    st.info(f"Predicted Yield: {predicted_yield[0]:.2f} hg/ha")

# Load and display image
image_path = os.path.join('images/SDG2.png')
st.image(image_path, caption='Zero Hunger', width=64)

# Create Streamlit app
st.title("Bumper Crop Insights Yield Prediction")

# Create input fields for user input
country = st.selectbox("Select Country", data['Area'].unique()) # Create dropdown list for country selection
items_for_selected_country = data[data['Area'] == country]['Item'].unique()
item = st.selectbox("Select Item", items_for_selected_country)  # Create dropdown list for item selection
pesticides = st.number_input("Pesticides (tonnes):")
avg_temp = st.number_input("Average Temperature (°C):")
rainfall = st.number_input("Rainfall (mm/year):")

# Button to predict yield
if st.button("Predict Yield"):
    predict_yield(country, item, pesticides, avg_temp, rainfall)
