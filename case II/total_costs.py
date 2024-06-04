import pandas as pd
import math

# Read the CSV files
service_points = pd.read_csv('service_points.csv', sep=';')
node_square_data = pd.read_csv('node_square_data.csv')
edges = pd.read_csv('edges.csv', sep=';')
predictions = pd.read_csv('predictions.csv')  # Read the predictions file

# Replace commas with dots and convert DIST column to numeric
edges['DIST'] = edges['DIST'].str.replace(',', '.').astype(float)

# Merge predictions with node_square_data to associate each node's predicted demand with its nearest service point
merged_data = pd.merge(predictions, node_square_data, left_on='node_id', right_on='node_id')

# Calculate the total predicted yearly demand for each service point
predicted_demand_per_point = merged_data.groupby('nearest_service_point')['Predicted'].sum().reset_index()

# Create a dictionary to store total road length for each service point
service_point_distances = {}

# Iterate over each service point
for service_point_id in node_square_data['nearest_service_point'].unique():
    service_point_nodes = set(
        node_square_data[node_square_data['nearest_service_point'] == service_point_id]['node_id'])
    total_distance = 0

    # Iterate over each edge and sum distances for edges within the service point's area
    for _, row in edges.iterrows():
        if row['V1'] in service_point_nodes and row['V2'] in service_point_nodes:
            total_distance += row['DIST']

    # Store the total distance in meters
    service_point_distances[service_point_id] = total_distance

# Calculate the yearly delivery cost for each service point using predicted demand
yearly_delivery_cost_per_point = []

for _, row in predicted_demand_per_point.iterrows():
    service_point_id = row['nearest_service_point']
    yearly_demand = row['Predicted']
    total_distance_meters = service_point_distances.get(service_point_id, 0)  # Get total distance in meters
    total_distance_km = total_distance_meters / 1000  # Convert to kilometers

    multiplier = max(math.ceil(yearly_demand / 36500), 1)  # Ensure at least one truck
    yearly_cost = total_distance_km * 0.5 * multiplier  # Annual cost calculation

    yearly_delivery_cost_per_point.append(yearly_cost)

# Total yearly delivery cost
total_yearly_delivery_cost = sum(yearly_delivery_cost_per_point)

# Calculate the number of service points
num_service_points = service_points.shape[0]

# Fixed yearly cost per service point
fixed_yearly_cost_per_point = 75000

# Total fixed yearly cost
total_fixed_yearly_cost = num_service_points * fixed_yearly_cost_per_point

# Calculate the maximum parcels picked up per day for each service point
parcels_picked_up = pd.read_csv('service_point_parcels_picked_up.csv', sep=';')
max_parcels_per_day = parcels_picked_up.drop('day\\location', axis=1).max()

# Calculate the daily capacity cost per service point
daily_capacity_cost_per_point = max_parcels_per_day * 0.1

# Calculate the yearly capacity cost per service point
yearly_capacity_cost_per_point = daily_capacity_cost_per_point * 365

# Total yearly capacity cost
total_yearly_capacity_cost = yearly_capacity_cost_per_point.sum()

# Total yearly cost (fixed + capacity + delivery)
total_yearly_cost = total_fixed_yearly_cost + total_yearly_capacity_cost + total_yearly_delivery_cost

print(f'Total yearly cost: ${total_yearly_cost:.2f}')
print('total distance:', total_distance_km)
print('predicted demand per point:', predicted_demand_per_point)
print('yearly demand:', yearly_demand)
print('predicted demand:', merged_data)



#change the distance to have a proportion for each node, based on the demand from linear regression. so each node
#will have demand and the ratio, and just calculate from that the delivery costs.
#add the variables from the long file to the nodes, based on node id (so avoid those that were deleted in the process)
#so instead of node_distances.csv use a file with node distance, node id, ratio and demand.
#change the cost calculation into a function which can be called
#USE THE DISTRIBUTION OF 80% AT 200M AND DROPPING TO 0% AT 2KM, which will be from regression by andrei (predicted demand
#and ratio for each node)
#do this: total road length for each service point (allocate roads with two different service points to both)
# * total deliveries predicted from regression (or something) assume capacity of a truck to 100, if lower then 100, then
#cover the road length once, if higher, like 150 then cover the road length twice.
