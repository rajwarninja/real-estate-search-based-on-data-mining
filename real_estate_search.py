import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

# Load dataset
data = pd.read_csv('filled_dataset.csv')

# Column mappings
categorical_features = ['city', 'state']  # Location-based features
numerical_features = ['house_size', 'bed', 'bath']  # Size and room-related features
target = 'price'

# Separate features and target
X = data.drop(columns=[target])
y = data[target]

# Preprocessing pipeline (no model training here)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Function for user search
def real_estate_search(city=None, state=None, min_budget=None, max_budget=None, min_bedrooms=None, max_bedrooms=None):
    # Filter data based on user input
    filtered_data = data.copy()
    
    # Apply filters based on user criteria
    if city:
        filtered_data = filtered_data[filtered_data['city'].str.contains(city, case=False)]
    if state:
        filtered_data = filtered_data[filtered_data['state'].str.contains(state, case=False)]
    if min_budget:
        filtered_data = filtered_data[filtered_data['price'] >= min_budget]
    if max_budget:
        filtered_data = filtered_data[filtered_data['price'] <= max_budget]
    if min_bedrooms:
        filtered_data = filtered_data[filtered_data['bed'] >= min_bedrooms]
    if max_bedrooms:
        filtered_data = filtered_data[filtered_data['bed'] <= max_bedrooms]
    
    # Check if filtered data is empty
    if filtered_data.empty:
        return "No properties found matching your criteria."
    
    # Cast 'bed' and 'bath' columns to integers before displaying
    filtered_data['bed'] = filtered_data['bed'].astype(int)
    filtered_data['bath'] = filtered_data['bath'].astype(int)
    
    # Return the filtered properties without predicting the price
    return filtered_data[['city', 'state', 'house_size', 'bed', 'bath', 'price']]

# User interface for search
print("Welcome to the Real Estate Search!")
user_city = input("Enter preferred city (or leave blank): ")
user_state = input("Enter preferred state (or leave blank): ")
user_min_budget = input("Enter minimum budget (or leave blank): ")
user_max_budget = input("Enter maximum budget (or leave blank): ")
user_min_bedrooms = input("Enter minimum number of bedrooms (or leave blank): ")
user_max_bedrooms = input("Enter maximum number of bedrooms (or leave blank): ")

# Convert inputs to appropriate types
user_min_budget = float(user_min_budget) if user_min_budget else None
user_max_budget = float(user_max_budget) if user_max_budget else None
user_min_bedrooms = int(user_min_bedrooms) if user_min_bedrooms else None
user_max_bedrooms = int(user_max_bedrooms) if user_max_bedrooms else None

# Perform search
results = real_estate_search(
    city=user_city,
    state=user_state,
    min_budget=user_min_budget,
    max_budget=user_max_budget,
    min_bedrooms=user_min_bedrooms,
    max_bedrooms=user_max_bedrooms
)

# Display results
print("\nSearch Results:")
print(results)
