import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
file_path = "house price data.csv"  # Ensure the file is in the same directory
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File not found. Please check the file path.")
    exit()

# Step 3: Inspect the dataset
print("\nFirst 5 rows of the dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# Step 4: Data preprocessing
# Remove null values
data = data.dropna()
print("\nData after dropping null values:")
print(data.info())

# Remove duplicate rows
data = data.drop_duplicates()
print("\nData after dropping duplicates:")
print(data.info())

# Step 5: Separate features (X) and target variable (y)
target_column = "price"  # Ensure this matches the dataset's column name
X = data[["sqft_living"]]  # Using 'sqft_living' as the single feature for simplicity
y = data[target_column]

# Step 6: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training (80%) and testing (20%) sets.")

# Step 7: Implement the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)  # Train the model
print("\nLinear Regression model trained successfully!")

# Step 8: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2 Score): {r2}")

# Step 9: Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted Regression Line")
plt.title("House Price Prediction")
plt.xlabel("Square Footage of Living Space")
plt.ylabel("House Price")
plt.legend()
plt.show()
