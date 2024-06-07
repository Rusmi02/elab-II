import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('allocations.csv')
df.drop(columns=['assigned_service_point','distance'], inplace=True)


# Read in the train_predictions.csv file
train_predictions_df = pd.read_csv('train_predictions.csv')
pickup_predictions_df = pd.read_csv('predictions_pickups.csv')

# Merge the two DataFrames based on node_id
merged_df = pd.merge(df, train_predictions_df, on='node_id', how='left')

# Read in the merged_data.csv file
merged_data_df = pd.read_csv('merged_data.csv')

# Select only the 'node_id' and 'assigned_population' columns from merged_data_df
merged_data_subset = merged_data_df[['node_id', 'assigned_population']]

# Merge the subset of merged_data_df with the previously merged DataFrame based on node_id
merged_df = pd.merge(merged_df, merged_data_subset, on='node_id', how='left')
merged_df = pd.merge(merged_df, pickup_predictions_df, on='node_id', how='left')

# Calculate the replacement value demand
# 10.899 is average demand per capita for current service points
# 0.5726 is the average pick up ratio for current service points
replacement_values_demand = merged_df['assigned_population'] * 10.899
replacement_values_pickup = 0.5726

# Replace NaN values in the 'predicted_demand' column with the calculated replacement values
merged_df['Predicted_Demand'].fillna(replacement_values_demand, inplace=True)
merged_df['Predicted'].fillna(replacement_values_pickup, inplace=True)

# Rename the 'predicted_demand' column to 'Pickup_Ratio'
merged_df.rename(columns={'Predicted': 'Pickup_Ratio'}, inplace=True)


print(merged_df.iloc[0])
print(len(merged_df))

# Output the final merged DataFrame to a new CSV file if needed
merged_df.to_csv('final_data.csv', index=False)