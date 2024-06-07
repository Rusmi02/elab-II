# cost calculation function - ideal inputs:
# array of service point locations - just node ids
# node_square_data -  used to get which service point is nearest to which node
# service parcel picked up  - replace with array of service point id + capacity,
# so currently it takes predicted deliveries from regression, and the capacity
# from what's given in the file -
# change instead to a distance based function - so each node has an expected demand,
# the assigned service point, and distance to that service point,
# then determine pickups and deliveries based on just that distance.
# node_demand - node_id, demand
# node_service_point - node_id, nearest_service_point_id, distance
# capacity_of_point - service_point_id, capacity

import pandas as pd


def calculate_yearly_cost(node_demand_file, node_service_point_file, capacity_of_point_file):
    # Read in the data
    node_demand = pd.read_csv(node_demand_file)
    node_service_point = pd.read_csv(node_service_point_file)
    capacity_of_point = pd.read_csv(capacity_of_point_file)

    # Merge node_demand and node_service_point on node_id
    node_data = pd.merge(node_demand, node_service_point, on='node_id')

    # Convert distance to km
    node_data['distance_km'] = node_data['distance'] / 1000

    # Calculate the percentage of deliveries based on distance
    def calculate_delivery_percentage(distance):
        if distance <= 200:
            return 0.2 * (distance / 200)
        elif distance <= 2000:
            return 0.2 + (distance - 200) / 1800 * 0.8
        else:
            return 1.0

    # Apply the delivery percentage calculation
    node_data['delivery_percentage'] = node_data['distance'].apply(calculate_delivery_percentage)

    # Calculate the yearly deliveries per node
    node_data['yearly_deliveries'] = node_data['Predicted_Demand'] * node_data['delivery_percentage']

    # Calculate the yearly delivery cost for each node
    node_data['yearly_delivery_cost'] = node_data['yearly_deliveries'] * node_data['distance_km'] * 0.5

    # Calculate the total yearly delivery cost for each service point
    yearly_delivery_cost_per_point = node_data.groupby('assigned_service_point')['yearly_delivery_cost'].sum()

    # Total yearly delivery cost
    total_yearly_delivery_cost = yearly_delivery_cost_per_point.sum()

    # Calculate the number of service points
    num_service_points = node_service_point['assigned_service_point'].nunique()

    # Fixed yearly cost per service point
    fixed_yearly_cost_per_point = 75000

    # Total fixed yearly cost
    total_fixed_yearly_cost = num_service_points * fixed_yearly_cost_per_point

    # Calculate the capacity cost for each service point
    capacity_of_point = capacity_of_point.set_index('service_point_id')
    daily_capacity_cost_per_point = capacity_of_point['capacity'] * 0.1

    # Calculate the yearly capacity cost per service point
    yearly_capacity_cost_per_point = daily_capacity_cost_per_point * 365

    # Total yearly capacity cost
    total_yearly_capacity_cost = yearly_capacity_cost_per_point.sum()

    # Total yearly cost (fixed + capacity + delivery)
    total_yearly_cost = total_fixed_yearly_cost + total_yearly_capacity_cost + total_yearly_delivery_cost

    return total_yearly_cost


# Example usage
node_demand_file = 'final_data.csv'
node_service_point_file = 'allocations.csv'
capacity_of_point_file = 'capacity_of_point.csv'

total_yearly_cost = calculate_yearly_cost(node_demand_file, node_service_point_file, capacity_of_point_file)

print(f'Total yearly cost: ${total_yearly_cost:.2f}')
