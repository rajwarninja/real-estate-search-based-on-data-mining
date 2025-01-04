import pandas as pd

# Path to the CSV file
file_path = 'dataset.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Display initial dataset info to check missing values
print("Initial dataset info:")
print(df.info())

# Filling missing values in numerical columns with the mean (or median if preferred)
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numerical_columns:
    mean_value = df[column].mean()
    df[column].fillna(mean_value, inplace=True)

# Filling missing values in categorical columns with the mode (most frequent value)
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    mode_value = df[column].mode()[0] 
    df[column].fillna(mode_value, inplace=True) 

# Display data info after filling missing values
print("\nDataset info after filling missing values:")
print(df.info())

# Optionally, inspect the first few rows after filling missing values
print("\nFirst few rows after filling missing values:")
print(df.head())

# Optional: Save the cleaned dataset to a new file
df.to_csv('filled_dataset.csv', index=False)
