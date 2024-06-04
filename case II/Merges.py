import pandas as pd

# Load the data
pop_assigned_df = pd.read_csv('pop_assigned_square-basis.csv')
modified_squares_df = pd.read_csv('modified_squares.csv')
service_points_df = pd.read_csv('service_points.csv', delimiter=';')
node_distances_df = pd.read_csv('node_distances.csv')

# Merge the population data with the modified squares data
merged_df = pop_assigned_df.merge(modified_squares_df, left_on='assigned_square', right_on='Square')

# Merge the service points data
merged_df = merged_df.merge(service_points_df, left_on='assigned_square', right_on='Square', suffixes=('', '_service'))

# Calculate the adjustment factor for each node
merged_df['adjustment_factor'] = merged_df['assigned_population'] / merged_df['Population']

# Columns to be adjusted based on population
cols_to_adjust = [
    'Male', 'Female', 'Age0-14', 'Age15-24', 'Age25-44', 'Age45-64', 'Age65+',
    'Households', 'Single-person households', 'Multi-person households w/o kids',
    'Single parent households', 'Two-parent households', 'Houses', 'Vacant houses'
]

# Adjust the values
for col in cols_to_adjust:
    merged_df[f'adjusted_{col}'] = merged_df[col] * merged_df['adjustment_factor']

# Keep the original ratios and averages
cols_to_keep = [
    'Home ownership %', 'Rental %', 'Social housing %', 'Avg. home value',
    'Urbanization index', 'Median household income', 'Percentage low income households',
    'Percentage high income households', 'Distance nearest supermarket in km', 'Male.prop',
    'Female.prop', 'Age0-14.prop', 'Age15-24.prop', 'Age25-44.prop', 'Age45-64.prop',
    'Age65+.prop', 'Single-person households.prop', 'Multi-person households w/o kids.prop',
    'Single parent households.prop', 'Two-parent households.prop', 'hold_house_ratio',
    'pop_house_ratio', 'pop_hold_ratio', 'Income_median'
]

# Adjust service points data based on the node's assigned population relative to the service point's total population
merged_df['adjusted_total_deliveries'] = merged_df['Total Deliveries'] * (merged_df['assigned_population'] / merged_df['Population_service'])
merged_df['adjusted_total_pickups'] = merged_df['Total Pickups'] * (merged_df['assigned_population'] / merged_df['Population_service'])
merged_df['adjusted_total_demand'] = merged_df['adjusted_total_deliveries'] + merged_df['adjusted_total_pickups']
merged_df['adjusted_pickup_ratio'] = merged_df['adjusted_total_pickups'] / merged_df['adjusted_total_demand']

# Select relevant columns for the final dataframe
final_cols = ['node_id', 'x', 'y', 'square', 'nearest_service_point', 'population', 'assigned_population', 'assigned_square'] + \
             [f'adjusted_{col}' for col in cols_to_adjust] + cols_to_keep + \
             ['adjusted_total_deliveries', 'adjusted_total_pickups', 'adjusted_total_demand', 'adjusted_pickup_ratio']

final_df = merged_df[final_cols]

# Filter out nodes with missing demographic data (e.g., Male = 0)
final_df = final_df[final_df['adjusted_Male'] != 0]

# Calculate the weighted average distance for each service point
merged_distances_df = final_df.merge(node_distances_df, left_on=['node_id', 'nearest_service_point'], right_on=['node_id', 'nearest_service_point'])
merged_distances_df['weighted_distance'] = merged_distances_df['assigned_population'] * merged_distances_df['distance_to_service_point']

weighted_avg_distance_df = merged_distances_df.groupby('nearest_service_point').apply(
    lambda x: x['weighted_distance'].sum() / x['assigned_population'].sum()
).reset_index()

weighted_avg_distance_df.columns = ['nearest_service_point', 'average_distance']

# Merge the weighted average distance back into the final dataframe
final_df = final_df.merge(weighted_avg_distance_df, on='nearest_service_point', how='left')

# Save the final dataframe to a new CSV file
final_df.to_csv('combined_data_with_service_points_filtered_with_avg_distance.csv', index=False)
