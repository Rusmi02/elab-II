import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
data = pd.read_csv('merged_data.csv')
data['Distance nearest supermarket in km'] = pd.to_numeric(data['Distance nearest supermarket in km'], errors='coerce')

# Step 2: Data Preprocessing
# Drop rows where square column has missing values (NA)
data = data.dropna(subset=['square_x'])
# Drop rows where "Income Median" column has missing values (NA)
data = data.dropna(subset=['Income_median'])

# Define the feature columns to use (excluding ratios)
features = [
    'adjusted_Male',
    'adjusted_Female', 'adjusted_Age0-14', 'adjusted_Age15-24', 'adjusted_Age25-44',
    'adjusted_Age45-64', 'adjusted_Age65+', 'Home ownership %', 'Rental %'
]

# Define the target variable
target = 'adjusted_total_demand'

# Ensure 'node_id' is present in the DataFrame
if 'node_id' not in data.columns:
    data['node_id'] = range(len(data))

# Step 3: Split the data into training and test sets (80-20 split)
X = data[features]
y = data[target]

num_rows = len(X)
train_size = int(num_rows)

# Split the data into training and test sets
X_train = X[:train_size]
X_test = X[train_size:]

# Add a constant to the model (intercept)
X_train = sm.add_constant(X_train)
y_train = y[:train_size]

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

# Coefficients of the model
coefficients = pd.DataFrame({'Feature': ['const'] + features, 'Coefficient': model.coef_})
print(coefficients)

# Summary of the model using statsmodels
model_significant = sm.OLS(y, sm.add_constant(X)).fit()
print(model_significant.summary())

# Step 6: Save the predicted and real values into a CSV file
train_results = pd.DataFrame({
    'node_id': data['node_id'][:train_size],
    'Actual_Demand': y_train,
    'Predicted_Demand': y_pred_train
})

train_results['Predicted_Demand'] = np.where(train_results['Predicted_Demand'] < 0, train_results['Actual_Demand'] / 2, train_results['Predicted_Demand'])
print(train_results.iloc[0])

train_results.drop(columns=['Actual_Demand'], inplace=True)

train_results.to_csv('train_predictions.csv', index=False)


print('Train predictions saved to train_predictions.csv')
