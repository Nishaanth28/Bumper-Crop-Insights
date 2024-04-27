import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

dataset_path = r"C:\Rpro\labtest\yield_df.csv"
df = pd.read_csv(dataset_path)

# Explore the dataset
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Handle missing values
df.dropna(inplace=True)

# Separate features (X) and target variable (y)
X = df.drop(['hg/ha_yield'], axis=1)  # Features
y = df['hg/ha_yield']  # Target variable

print("Dataset information:")
print(df.info())

# Define columns for one-hot encoding and standardization
categorical_cols = ['Area', 'Item']
numeric_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing of training data
print("\nPreprocessing training data:")
try:
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    print("Preprocessed training data:")
    print(X_train_preprocessed[:5])  # Print first 5 rows of preprocessed data
except Exception as e:
    print("Error occurred during preprocessing of training data:", e)

# Preprocessing of testing data
print("\nPreprocessing testing data:")
try:
    X_test_preprocessed = preprocessor.transform(X_test)
    print("Preprocessed testing data:")
    print(X_test_preprocessed[:5])  # Print first 5 rows of preprocessed data
except Exception as e:
    print("Error occurred during preprocessing of testing data:", e)

# Identify columns with non-numeric values
non_numeric_columns = df.select_dtypes(exclude=['float64', 'int64']).columns
print("Columns with non-numeric values:", non_numeric_columns)

# Print non-numeric values in each column
for col in non_numeric_columns:
    print(f"Non-numeric values in column '{col}':")
    print(df[col][df[col].apply(lambda x: isinstance(x, str))])

