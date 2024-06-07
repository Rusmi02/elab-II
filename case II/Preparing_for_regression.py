import pandas as pd
import csv

# Set the maximum number of rows and columns to display
pd.set_option('display.max_rows', None)    # To display all rows
pd.set_option('display.max_columns', None)  # To display all columns

# Step 1: Load the CSV files into DataFrames
pop_assigned = pd.read_csv("pop_assigned_square-basis.csv")
modified_squares = pd.read_csv("modified_squares.csv")

# Step 2: Remove the column assigned_square_y from pop_assigned
pop_assigned.drop(columns=['assigned_square_y'], inplace=True)

# Step 3: Merge the two DataFrames based on assigned_square field in pop_assigned
# and Square field in modified_squares
merged_df = pd.merge(pop_assigned, modified_squares, left_on='assigned_square_x', right_on='Square', how='inner')

# Define the columns for which you want to create adjusted columns
columns_to_adjust = ['Male', 'Female', 'Age0-14', 'Age15-24', 'Age25-44', 'Age45-64', 'Age65+']

# Create adjusted columns
for column in columns_to_adjust:
    adjusted_column_name = f"adjusted_{column}"
    merged_df[adjusted_column_name] = merged_df[column] * (merged_df['assigned_population'] / merged_df['population'])

# merging to provide target:

import pandas as pd

def read_service_points(file_path):
    service_points = {}
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        next(reader)
        for row in reader:
            sp_id, x, y, square, population, total_deliveries, total_pickups = row
            service_points[int(sp_id)] = {
                "service_point_id": int(sp_id),
                "x": float(x),
                "y": float(y),
                "square": str(square),
                "population": int(population),
                "total_deliveries": int(total_deliveries),
                "total_pickups": int(total_pickups),
            }
    return service_points

# Load service point data into a DataFrame
service_points_data = read_service_points("service_points.csv")
service_points_df = pd.DataFrame.from_dict(service_points_data, orient='index')

# Merge with merged_df
merged_with_service_points_df = pd.merge(merged_df, service_points_df, left_on='assigned_service_point', right_on='service_point_id', how='inner')

# Define the columns for which you want to create adjusted columns
columns_to_adjust = ['total_deliveries', 'total_pickups']

# Create a mapping dictionary from assigned_service_point to population
service_point_population_mapping = service_points_df['population'].to_dict()

# Create adjusted columns
for column in columns_to_adjust:
    adjusted_column_name = f"adjusted_{column}"
    merged_with_service_points_df[adjusted_column_name] = merged_with_service_points_df[column] * (merged_with_service_points_df['assigned_population'] / merged_with_service_points_df['assigned_service_point'].map(service_point_population_mapping))

# Calculate total_demand and adjusted_total_demand
merged_with_service_points_df['total_demand'] = merged_with_service_points_df['total_deliveries'] + merged_with_service_points_df['total_pickups']
merged_with_service_points_df['adjusted_total_demand'] = merged_with_service_points_df['adjusted_total_deliveries'] + merged_with_service_points_df['adjusted_total_pickups']
merged_with_service_points_df['pickup_ratio'] = merged_with_service_points_df['total_pickups']/ merged_with_service_points_df['total_deliveries']

# Print the full first row of the dataset
print(merged_with_service_points_df.iloc[0])

# Step 4: Save the merged DataFrame to a new CSV file
merged_with_service_points_df.to_csv("merged_data.csv", index=False)

#Step 5: find node ids for which won't predict demand directly - nstead will multiply pop by 10.899 - average total demand per person per year from servic point locations

# Find the rows where square_x is NA or population_x is 2
special_nodes_df = merged_with_service_points_df[(merged_with_service_points_df['square_x'].isna()) | (merged_with_service_points_df['population_x'] == 2)]

# Select the node_id for those rows
special_nodes = special_nodes_df['node_id']

# Save the node_id into a CSV file called special_nodes.csv
special_nodes.to_csv('special_nodes.csv', index=False)

