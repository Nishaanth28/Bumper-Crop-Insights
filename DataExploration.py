from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = r"C:\Rpro\labtest\yield_df.csv"
df = pd.read_csv(dataset_path)

# Display basic information about the dataset
print("Dataset information:")
print(df.info())

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Set the aesthetic style of the plots
sns.set_style("whitegrid")

# Increase the default figure size for better readability
plt.rcParams['figure.figsize'] = [10, 6]



# Visualizations
# Histogram of yield
plt.figure(figsize=(8, 6))
sns.histplot(df['hg/ha_yield'], bins=10, kde=True)
plt.title('Histogram of Yield')
plt.xlabel('Yield')
plt.ylabel('Frequency')
plt.show()


# Scatter plot of yield vs. average rainfall
plt.figure(figsize=(8, 6))
sns.scatterplot(x='average_rain_fall_mm_per_year', y='hg/ha_yield', data=df)
plt.title('Yield vs. Average Rainfall')
plt.xlabel('Average Rainfall (mm/year)')
plt.ylabel('Yield (hg/ha)')
plt.show()

# Box plot of yield by year
plt.figure(figsize=(10, 6))
sns.boxplot(x='Year', y='hg/ha_yield', data=df)
plt.title('Yield by Year')
plt.xlabel('Year')
plt.ylabel('Yield (hg/ha)')
plt.xticks(rotation=45)
plt.show()

numeric_columns = ['Year', 'hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
numeric_df = df[numeric_columns]

# Correlation matrix
print("\nCorrelation matrix:")
correlation_matrix = numeric_df.corr()
print(correlation_matrix)
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Scatter plots for numeric columns
numeric_columns = ['hg/ha_yield', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=col, y='hg/ha_yield')
    plt.title(f"hg/ha_yield vs {col}")
    plt.xlabel(col)
    plt.ylabel("hg/ha_yield")
    plt.show()

    ## Area (Bar Plot of Top 10 Areas)
    plt.figure()
    area_counts = df['Area'].value_counts().head(10)
    sns.barplot(x=area_counts.values, y=area_counts.index, palette='coolwarm')
    plt.title('Top 10 Areas')
    plt.xlabel('Count')
    plt.ylabel('Area')
    plt.show()

    ## Item (Bar Plot of Items)
    plt.figure()
    item_counts = df['Item'].value_counts()
    sns.barplot(x=item_counts.values, y=item_counts.index, palette='viridis')
    plt.title('Items Count')
    plt.xlabel('Count')
    plt.ylabel('Item')
    plt.show()

    ## Year (Line Plot of Entries Over Years)
    plt.figure()
    year_counts = df['Year'].value_counts().sort_index()
    sns.lineplot(x=year_counts.index, y=year_counts.values, marker='o', color='b')
    plt.title('Entries Over Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Entries')
    plt.show()

    ## hg/ha_yield (Histogram)
    plt.figure()
    sns.histplot(df['hg/ha_yield'], bins=30, kde=True, color='g')
    plt.title('Distribution of hg/ha_yield')
    plt.xlabel('hg/ha_yield')
    plt.ylabel('Frequency')
    plt.show()

    ## average_rain_fall_mm_per_year (Histogram)
    plt.figure()
    sns.histplot(df['average_rain_fall_mm_per_year'], bins=30, kde=True, color='y')
    plt.title('Distribution of Average Rainfall (mm/year)')
    plt.xlabel('Average Rainfall (mm/year)')
    plt.ylabel('Frequency')
    plt.show()

    ## pesticides_tonnes (Histogram)
    plt.figure()
    sns.histplot(df['pesticides_tonnes'], bins=30, kde=True, color='m')
    plt.title('Distribution of Pesticides (tonnes)')
    plt.xlabel('Pesticides (tonnes)')
    plt.ylabel('Frequency')
    plt.show()

    ## avg_temp (Histogram)
    plt.figure()
    sns.histplot(df['avg_temp'], bins=30, kde=True, color='c')
    plt.title('Distribution of Average Temperature')
    plt.xlabel('Average Temperature (°C)')
    plt.ylabel('Frequency')
    plt.show()

    # Select only numeric columns
    numeric_columns = df.select_dtypes(include='number')

    # Calculate the correlation matrix
    corr_matrix = numeric_columns.corr()

    # Create a heatmap to visualize the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap of Dataset')
    plt.show()

