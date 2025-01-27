# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx
import plotly.figure_factory as ff
import matplotlib.patches as patches

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

# Convert 'date' column to datetime format (if applicable)
if 'date' in data.columns:
    data['date'] = pd.to_datetime(data['date'], errors='coerce')  # Coerce errors to NaT
    data['year'] = data['date'].dt.year
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day

# Remove null values
data = data.dropna()

# Remove duplicate rows
data = data.drop_duplicates()

# Step 5: Remove non-numeric columns (such as addresses)
# Here we check for numeric columns and exclude non-numeric ones
numeric_columns = data.select_dtypes(include=[np.number]).columns
data = data[numeric_columns]  # Keep only numeric columns

# Step 6: Separate features (X) and target variable (y)
target_column = "price"  # Ensure this matches the dataset's column name
X = data[["sqft_living"]]  # Using 'sqft_living' as the single feature for simplicity
y = data[target_column]

# Step 7: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nData split into training (80%) and testing (20%) sets.")

# Step 8: Implement the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)  # Train the model
print("\nLinear Regression model trained successfully!")
plt.show()
# Step 9: Generate all types of visualizations
# 1. Line Graph
plt.figure(figsize=(10, 6))
plt.plot(X_test, y_pred, color="red", label="Regression Line")
plt.xlabel("Square Footage")
plt.ylabel("House Price")
plt.title("House Price Prediction - Line Graph")
plt.legend()
plt.show()

# 2. Bar Graph (Comparison of Actual vs Predicted Prices)
plt.figure(figsize=(10, 6))
indices = range(10)  # Select the first 10 data points from the test set
plt.bar(indices, y_test.iloc[:10], width=0.4, label='Actual Prices', color='blue', align='center')
plt.bar(indices, y_pred[:10], width=0.4, label='Predicted Prices', color='red', align='edge')
plt.title('Comparison of Actual vs Predicted House Prices')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.legend()
plt.show()

# 3. Pie Chart (Distribution of Predicted Prices as "High" vs "Low")
threshold_price = 500000  # Example threshold price
predicted_labels = ['High' if price > threshold_price else 'Low' for price in y_pred]

# Calculate the distribution
labels = ['High', 'Low']
sizes = [predicted_labels.count('High'), predicted_labels.count('Low')]
colors = ['#ff9999', '#66b3ff']
explode = (0.1, 0)  # "explode" the High slice

plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode, shadow=True)
plt.title('Distribution of Predicted House Prices (High vs Low)')
plt.show()

# 4. Histogram (Distribution of Actual vs Predicted House Prices)
plt.figure(figsize=(10, 6))
plt.hist(y_test, bins=20, alpha=0.5, label="Actual Prices")
plt.hist(y_pred, bins=20, alpha=0.5, label="Predicted Prices")
plt.legend()
plt.title("Distribution of Actual vs Predicted House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# 5. Scatter Plot (Actual vs Predicted House Prices)
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="blue", label="Actual Prices")
plt.scatter(X_test, y_pred, color="red", label="Predicted Prices")
plt.title("Actual vs Predicted House Prices - Scatter Plot")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.legend()
plt.show()

# 6. Area Chart (Cumulative predicted prices)
plt.figure(figsize=(10, 6))
plt.fill_between(X_test.values.flatten(), y_pred, color="red", alpha=0.3, label="Predicted Prices")
plt.title("Area Chart - Predicted House Prices")
plt.xlabel("Square Footage")
plt.ylabel("House Price")
plt.legend()
plt.show()

# 7. Bubble Chart (Actual vs Predicted with Bubble Size for a feature, e.g., Number of Bedrooms)
# Assuming you have a "bedrooms" column in the data
bubble_size = np.random.randint(50, 200, size=len(X_test))

plt.figure(figsize=(10, 6))
plt.scatter(X_test.values.flatten(), y_test, s=bubble_size, color="blue", alpha=0.5, label="Actual Prices")
plt.scatter(X_test.values.flatten(), y_pred, s=bubble_size, color="red", alpha=0.5, label="Predicted Prices")
plt.title("Bubble Chart of House Prices")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.legend()
plt.show()

# 8. Heatmap (Correlation between Features)
correlation_matrix = data.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 9. Treemap (Price distribution by Year, as 'zipcode' might not exist)
if 'year' in data.columns:
    fig = px.treemap(data, path=["year"], values="price", title="Price Distribution by Year")
    fig.show()
else:
    print("The 'year' column is not present in the dataset. Please choose another feature.")

# 10. Box Plot (Boxplot of House Prices)
plt.figure(figsize=(10, 6))
sns.boxplot(y=data['price'])
plt.title("Box Plot of House Prices")
plt.show()

# 11. Violin Plot
plt.figure(figsize=(10, 6))
sns.violinplot(y=data['price'])
plt.title("Violin Plot of House Prices")
plt.show()

# 12. Waterfall Chart (Plotly)
fig = go.Figure(go.Waterfall(
    y=[100, -50, 60, -30, 80],
    base=0,
    measure=["absolute", "relative", "relative", "relative", "relative"],
    name="Waterfall Example"
))
fig.show()

# 13. Stacked Chart
plt.figure(figsize=(10, 6))
df = pd.DataFrame({
    'category1': [50, 60, 70],
    'category2': [30, 40, 50]
}, index=["A", "B", "C"])
df.plot(kind='bar', stacked=True)
plt.title("Stacked Bar Chart")
plt.show()

# 14. Radar Chart
labels = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']
stats = [20, 34, 30, 35]
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
ax.plot(labels, stats, linewidth=2, linestyle='solid')
ax.fill(labels, stats, alpha=0.4)
plt.title("Radar Chart Example")
plt.show()

# 15. Gauge Chart (Plotly)
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=72,
    title={'text': "Gauge Chart Example"},
    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}}))
fig.show()

# 16. 3D Graph
fig = px.scatter_3d(data, x='sqft_living', y='price', z='bedrooms', title="3D Scatter Plot")
fig.show()

# 17. Interactive Graph
fig = px.scatter(data, x="sqft_living", y="price", title="Interactive Scatter Plot")
fig.show()

# 18. Dynamic Graph
fig = px.scatter(data, x="sqft_living", y="price", animation_frame="year", title="Dynamic Scatter Plot")
fig.show()

# 19. Network Graph
G = nx.erdos_renyi_graph(50, 0.1)
nx.draw(G, with_labels=True, node_size=500, font_size=10, font_color='white')
plt.title("Network Graph")
plt.show()

# 20. Flowchart (use diagrams library)
fig, ax = plt.subplots(figsize=(10, 6))
ax.add_patch(patches.FancyBboxPatch((0.1, 0.5), 0.3, 0.2, boxstyle="round,pad=0.1", facecolor="skyblue"))
ax.text(0.25, 0.6, "Start", horizontalalignment='center', verticalalignment='center', fontsize=12)
plt.axis("off")
plt.show()

# 21. Mind Map (Basic Tree Structure Example)
# You can use libraries like 'graphviz' for more complex mind maps.

# 22. Organizational Chart (Simple Example with Plotly)
org_chart = {
    "name": "CEO", "children": [
        {"name": "CTO", "children": [{"name": "Dev Team"}]},
        {"name": "CFO", "children": [{"name": "Finance Team"}]}
    ]
}
fig = px.sunburst(data_frame=[org_chart], path=["name"])
fig.show()

# 23. Gantt Chart (Plotly)
gantt_data = [
    dict(Task="Task 1", Start="2022-01-01", Finish="2022-01-10", Resource="Team A"),
    dict(Task="Task 2", Start="2022-01-11", Finish="2022-01-15", Resource="Team B")
]
fig = ff.create_gantt(gantt_data)
fig.show()

# 24. PERT Chart
# Plotly, graphviz, or other tools can be used to generate PERT charts based on relationships and dependencies.

# 25. Sankey Diagram
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=["Start", "Process A", "Process B", "End"]
    ),
    link=dict(
        source=[0, 1, 0],
        target=[1, 2, 3],
        value=[8, 4, 2]
    )
))
fig.show()
