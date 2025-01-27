# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the Dataset
file_path = "house price data.csv"  # Ensure the CSV file is in the same directory or provide full path
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# Step 3: Inspect the Dataset
print("\nFirst 5 rows of the dataset:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# Step 4: Data Preprocessing
# Check and remove null values
data = data.dropna()
print("\nData after dropping null values:")
print(data.info())

# Check and remove duplicates
data = data.drop_duplicates()
print("\nData after removing duplicates:")
print(data.info())

# Separate features (X) and target variable (y)
target_column = "price"  # Correct target column name based on dataset
X = data.drop(columns=[target_column])  # Features (all columns except the target)
y = data[target_column]  # Target
