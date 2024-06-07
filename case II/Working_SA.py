import pandas as pd
import random
import csv
import numpy as np
import math
import scipy.stats as stats


### SET UP METHODS - MOST TAKEN FROM PREVIOUS FILES:

##### To read in both distance csv correctly

### something round 300 for capacity

#### for implementing bounce back - take average st.dev as demand,
# take average per day for that point as the daily mean, build distribution based on that.

### this one shoudl be more correct but yeah
def allocate_nodes_to_points(distances, service_points_node_ids, node_id_to_index):
    allocations = []

    # Convert service points node IDs to their corresponding indices
    service_point_indices = [node_id_to_index[sp] for sp in service_points_node_ids]

    # Filter distances to only access rows corresponding to service points
    filtered_distances = distances[service_point_indices, :]

    # Iterate over each column in filtered distances
    for col_index in range(filtered_distances.shape[1]):
        # Find the node id corresponding to the index of this column
        node_id = list(node_id_to_index.keys())[col_index]

        # Find the index of the row where the smallest distance was found
        assigned_sp_index = np.argmin(filtered_distances[:, col_index])

        # Find the node id that maps to the index of the row with the smallest distance
        assigned_service_point = service_points_node_ids[assigned_sp_index]

        # Find the smallest distance for this column
        distance = np.min(filtered_distances[:, col_index])

        # Append the tuple to the allocations list
        allocations.append((node_id, assigned_service_point, distance))

    return allocations

# in case want to change allocation definition
def allocate_nodes_to_points2(distances, service_points_node_ids, node_id_to_index):
    allocations = []

    # Convert service points node IDs to their corresponding indices
    service_point_indices = [node_id_to_index[sp] for sp in service_points_node_ids]

    # Transpose distances so that service point ids are in the columns
    transposed_distances = distances.T

    # Filter transposed distances to only access columns corresponding to service points
    filtered_distances = transposed_distances[service_point_indices, :]

    # Iterate over each column in filtered distances
    for col_index in range(filtered_distances.shape[1]):
        # Find the node id corresponding to the index of this row
        node_id = list(node_id_to_index.keys())[col_index]

        # Find the index of the column where the smallest distance was found
        assigned_sp_index = np.argmin(filtered_distances[:, col_index])

        # Find the node id that maps to the index of the row with the smallest distance
        assigned_service_point = service_points_node_ids[assigned_sp_index]

        # Find the smallest distance for this row
        distance = np.min(filtered_distances[:, col_index])

        # Append the tuple to the allocations list
        allocations.append((node_id, assigned_service_point, distance))

    return allocations

# consider getting capacity at just the end, or changing it,
def calculate_yearly_cost_final(demand_data, allocations):
    # Read in the data
    node_demand = pd.DataFrame(demand_data)  # Assuming demand_data is already a DataFrame
    node_service_point = pd.DataFrame(allocations, columns=['node_id', 'nearest_service_point_id', 'distance'])

    # Convert distance to km
    node_service_point['distance_km'] = node_service_point['distance'] / 1000

    # Calculate the percentage of deliveries based on distance

    #
    def calculate_delivery_percentage(distance):
        if distance <= 200:
            return 0.2 * (distance / 200)
        elif distance <= 2000:
            return 0.2 + (distance - 200) / 1800 * 0.8
        else:
            return 1.0

    # Apply the delivery percentage calculation
    node_service_point['delivery_percentage'] = node_service_point['distance'].apply(calculate_delivery_percentage)

    # Merge node_demand and node_service_point on node_id
    node_data = pd.merge(node_demand, node_service_point, on='node_id')

    # Use 'Predicted_Demand' for demand
    node_data['yearly_deliveries'] = 0.9*(node_data['Predicted_Demand'] * node_data['delivery_percentage'])+0.1*node_data['delivery']

    # Calculate the yearly delivery cost for each node
    node_data['yearly_delivery_cost'] = node_data['yearly_deliveries'] * node_data['distance_km'] * 0.5

    # Calculate the total yearly delivery cost for each service point
    yearly_delivery_cost_per_point = node_data.groupby('nearest_service_point_id')['yearly_delivery_cost'].sum()

    # Total yearly delivery cost
    total_yearly_delivery_cost = yearly_delivery_cost_per_point.sum()

    # Calculate the number of service points
    num_service_points = node_service_point['nearest_service_point_id'].nunique()

    # Fixed yearly cost per service point
    fixed_yearly_cost_per_point = 75000

    # Total fixed yearly cost
    total_fixed_yearly_cost = num_service_points * fixed_yearly_cost_per_point

    node_data['yearly_pickup'] = 0.9 * (node_data['Predicted_Demand'] * (1-node_data['delivery_percentage'])) + 0.1 * \
                                     node_data['pickup']
    # will have to change the capacity somehow:
    # Calculate the capacity cost for each service point. Capacity obtained from a normal distribution, where the mean is
    # total yearly pickups / 312 (open days), and the st_dev is 0.500586532 of that - from historial data analysis:

    # Group the data by 'nearest_service_point_id' and calculate the sum of 'yearly_pickup'
    grouped_data = node_data.groupby('nearest_service_point_id')['yearly_pickup'].sum()

    # Define the standard deviation percentage
    std_dev_percentage = 65.90887727978011/100

    # Calculate mean capacities
    mean_capacity = grouped_data / 312

    # Calculate standard deviations
    std_dev_capacity = mean_capacity * std_dev_percentage

    # Find the z-value for the left tail of the distribution being 0.99
    z_value = stats.norm.ppf(0.995)

    # Calculate the capacity values
    capacity_values = mean_capacity + z_value * std_dev_capacity

    capacity_values_rounded_up = np.ceil(capacity_values)

    # Calculate the capacity costs
    capacity_costs = capacity_values_rounded_up * 0.1 * 312

    capacity_values_df = capacity_values_rounded_up.reset_index()
    capacity_values_df.columns = ['service_point_id', 'capacity_value']

    # Print service_point_id and its capacity value (rounded up)
    for index, row in capacity_values_df.iterrows():
        print(f"Service Point ID: {row['service_point_id']}, Capacity Value (rounded up): {row['capacity_value']}")

    # Calculate the total yearly capacity cost
    total_yearly_capacity_cost = capacity_costs.sum()

    # Total yearly cost (fixed + capacity + delivery)
    total_yearly_cost = total_fixed_yearly_cost + total_yearly_capacity_cost + total_yearly_delivery_cost

    print("Total capacity cost: " + str(total_yearly_capacity_cost))
    print("Total fixed cost: " + str(total_fixed_yearly_cost))
    print("Total delivery cost: " + str(total_yearly_delivery_cost))

    return total_yearly_cost

def calculate_yearly_cost(demand_data, allocations):
    # Read in the data
    node_demand = pd.DataFrame(demand_data)  # Assuming demand_data is already a DataFrame
    node_service_point = pd.DataFrame(allocations, columns=['node_id', 'nearest_service_point_id', 'distance'])

    # Convert distance to km
    node_service_point['distance_km'] = node_service_point['distance'] / 1000

    # Calculate the percentage of deliveries based on distance

    #
    def calculate_delivery_percentage(distance):
        if distance <= 200:
            return 0.2 * (distance / 200)
        elif distance <= 2000:
            return 0.2 + (distance - 200) / 1800 * 0.8
        else:
            return 1.0

    # Apply the delivery percentage calculation
    node_service_point['delivery_percentage'] = node_service_point['distance'].apply(calculate_delivery_percentage)

    # Merge node_demand and node_service_point on node_id
    node_data = pd.merge(node_demand, node_service_point, on='node_id')

    # Use 'Predicted_Demand' for demand
    node_data['yearly_deliveries'] = 0.9*(node_data['Predicted_Demand'] * node_data['delivery_percentage'])+0.1*node_data['delivery']

    # Calculate the yearly delivery cost for each node
    node_data['yearly_delivery_cost'] = node_data['yearly_deliveries'] * node_data['distance_km'] * 0.5

    # Calculate the total yearly delivery cost for each service point
    yearly_delivery_cost_per_point = node_data.groupby('nearest_service_point_id')['yearly_delivery_cost'].sum()

    # Total yearly delivery cost
    total_yearly_delivery_cost = yearly_delivery_cost_per_point.sum()

    # Calculate the number of service points
    num_service_points = node_service_point['nearest_service_point_id'].nunique()

    # Fixed yearly cost per service point
    fixed_yearly_cost_per_point = 75000

    # Total fixed yearly cost
    total_fixed_yearly_cost = num_service_points * fixed_yearly_cost_per_point

    node_data['yearly_pickup'] = 0.9 * (node_data['Predicted_Demand'] * (1-node_data['delivery_percentage'])) + 0.1 * \
                                     node_data['pickup']
    # will have to change the capacity somehow:
    # Calculate the capacity cost for each service point. Capacity obtained from a normal distribution, where the mean is
    # total yearly pickups / 312 (open days), and the st_dev is 0.500586532 of that - from historial data analysis:

    # Group the data by 'nearest_service_point_id' and calculate the sum of 'yearly_pickup'
    grouped_data = node_data.groupby('nearest_service_point_id')['yearly_pickup'].sum()

    # Define the standard deviation percentage
    std_dev_percentage = 85.90887727978011/100

    # Calculate mean capacities
    mean_capacity = grouped_data / 312

    # Calculate standard deviations
    std_dev_capacity = mean_capacity * std_dev_percentage

    # Find the z-value for the left tail of the distribution being 0.99
    z_value = stats.norm.ppf(0.995)

    # Calculate the capacity values
    capacity_values = mean_capacity + z_value * std_dev_capacity

    capacity_values_rounded_up = np.ceil(capacity_values)

    # Calculate the capacity costs
    capacity_costs = capacity_values_rounded_up * 0.1 * 312

    # Calculate the total yearly capacity cost
    total_yearly_capacity_cost = capacity_costs.sum()

    # Total yearly cost (fixed + capacity + delivery)
    total_yearly_cost = total_fixed_yearly_cost + total_yearly_capacity_cost + total_yearly_delivery_cost


    return total_yearly_cost

####### OPERATORS:

def add_service_point(nodes, service_points):
    """
    Adds a new service point by randomly selecting a node that is not already a service point.

    Parameters:
        nodes (list): List of node ids.
        service_points (set): Set of node ids representing current service points.

    Returns:
        new_service_point (str): Node id of the new service point.
    """
    # Filter nodes that are not already service points
    eligible_nodes = [node for node in nodes if node not in service_points]

    # Randomly select a node from the eligible nodes
    new_service_point = random.choice(eligible_nodes)

    # Add the new service point to the set of service points
    new_service_points = service_points.copy()
    new_service_points.append(new_service_point)

    return new_service_points

def remove_service_point(service_points):
    """
    Removes a service point if there is more than one service point open.

    Parameters:
        service_points (set): Set of node ids representing current service points.

    Returns:
        new_service_points (set): Updated set of node ids representing the service points after removing one service point.
    """
    # Ensure there is more than one service point open
    if len(service_points) > 1:
        # Randomly select a service point to remove
        service_point_to_remove = random.choice(list(service_points))

        # Create a copy of service_points and remove the selected service point
        new_service_points = service_points.copy()
        new_service_points.remove(service_point_to_remove)

        return new_service_points
    else:
        # If there is only one service point, return the current set of service points unchanged
        return service_points

def accept(current_cost, new_cost, temperature):
    if new_cost < current_cost:
        return True
    else:
        rand_num = random.random()
        return math.exp((current_cost - new_cost) / temperature) > rand_num


#####

### READ IN DATA

demand_data = pd.read_csv('final_data.csv')
# Calculate 'pickup' and 'delivery' columns
demand_data['pickup'] = demand_data['Predicted_Demand'] * demand_data['Pickup_Ratio']
demand_data['delivery'] = demand_data['Predicted_Demand'] * (1 - demand_data['Pickup_Ratio'])

nodes = demand_data['node_id'].tolist()

# Array that combines directed and unidrected distances, taking undirected distances whenever they are under 292.6m, and directed otherwise
distances = np.load('usable_distances.npy')

# map node id to index as theyre not always the same:

node_id_to_index = {node_id: index for index, node_id in enumerate(nodes)}


#### STARTING SOLUTION

random_node_id = random.choice(demand_data['node_id'].tolist())
service_points = [random_node_id]


allocations = allocate_nodes_to_points(distances, service_points, node_id_to_index)

cost = calculate_yearly_cost(demand_data, allocations)

##### Actual SA

## parameters
max_iterations = 10

# probabilities of operators
add_prob = 0.5

for iteration in range(1, max_iterations + 1):
    # Your code for each iteration goes here

    #Reset for new iteration, later add bounceback rate
    new_cost = cost
    new_service_points = service_points
    new_allocations = allocations

    temperature = max_iterations / iteration

    prob = random.random()

    if prob < add_prob:
        new_service_points = add_service_point(nodes, service_points)

        new_allocations = allocate_nodes_to_points(distances, new_service_points, node_id_to_index)
        new_cost = calculate_yearly_cost(demand_data, new_allocations)

        if accept(cost, new_cost, temperature):
            cost = new_cost
            service_points = new_service_points
            allocations = new_allocations
    else:
        new_service_points = remove_service_point(service_points)
        if( len(new_service_points) != len(service_points)):
            new_allocations = allocate_nodes_to_points(distances, new_service_points, node_id_to_index)
            new_cost = calculate_yearly_cost(demand_data, new_allocations)

            if accept(cost, new_cost, temperature):
                cost = new_cost
                service_points = new_service_points
                allocations = new_allocations


final_cost = calculate_yearly_cost_final(demand_data, allocations)
print(cost)




