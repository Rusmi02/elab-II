import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
data = pd.read_csv('combined_data_with_service_points_filtered_with_avg_distance.csv')
data['Distance nearest supermarket in km'] = pd.to_numeric(data['Distance nearest supermarket in km'], errors='coerce')

# Step 2: Data Preprocessing
# Remove rows with 'assigned_population' less than 0.5
data = data[data['assigned_population'] >= 0.5]

# Define the feature columns to use (excluding ratios)
features = [
    'assigned_population',
    'adjusted_Female',
    'adjusted_Age25-44',
    'adjusted_Age65+',
    'adjusted_Houses',
    'Home ownership %',
    'average_distance'
]

# Define the target variable
target = 'adjusted_total_deliveries'

# Include node_id for saving predictions
node_ids = data['node_id']

# Step 3: Split the data into training and test sets (80-20 split)
X = data[features]
y = data[target]

num_rows = len(X)
train_size = int(num_rows * 0.8)

# Split the data into training and test sets
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# Add a constant to the model (intercept)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model on the training set
y_pred_train = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_r2 = r2_score(y_train, y_pred_train)
train_percentage_difference = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100

print(f'Train RMSE: {train_rmse}')
print(f'Train R^2: {train_r2}')
print(f'Train Percentage Difference: {train_percentage_difference:.2f}%')

# Evaluate the model on the test set
y_pred_test = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_r2 = r2_score(y_test, y_pred_test)
test_percentage_difference = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

print(f'Test RMSE: {test_rmse}')
print(f'Test R^2: {test_r2}')
print(f'Test Percentage Difference: {test_percentage_difference:.2f}%')

# Print the first few actual and predicted values for inspection
print("First few actual vs predicted values (Train):")
print(pd.DataFrame({'Actual': y_train[:10], 'Predicted': y_pred_train[:10]}))

print("First few actual vs predicted values (Test):")
print(pd.DataFrame({'Actual': y_test[:10], 'Predicted': y_pred_test[:10]}))

# Coefficients of the model
coefficients = pd.DataFrame({'Feature': ['const'] + features, 'Coefficient': model.coef_})
print(coefficients)

# Summary of the model using statsmodels
model_significant = sm.OLS(y, sm.add_constant(X)).fit()
print(model_significant.summary())

# Combine predictions with node_ids for saving
results = pd.DataFrame({
    'node_id': node_ids,
    'Actual': np.concatenate([y_train, y_test]),
    'Predicted': np.concatenate([y_pred_train, y_pred_test])
})

# Save the results to a CSV file
results.to_csv('predictions.csv', index=False)

print("Predictions saved to 'predictions.csv'")

