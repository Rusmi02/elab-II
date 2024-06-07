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

    # Calculate the number of service points
    num_service_points = node_service_point['assigned_service_point'].nunique()

    # Fixed yearly cost per service point
    fixed_yearly_cost_per_point = 75000

    # Calculate the capacity cost for each service point
    capacity_of_point = capacity_of_point.set_index('service_point_id')
    daily_capacity_cost_per_point = capacity_of_point['capacity'] * 0.1

    # Calculate the yearly capacity cost per service point
    yearly_capacity_cost_per_point = daily_capacity_cost_per_point * 365

    # Combine all costs into a dataframe
    service_point_costs = pd.DataFrame({
        'yearly_delivery_cost': yearly_delivery_cost_per_point,
        'yearly_capacity_cost': yearly_capacity_cost_per_point,
        'fixed_yearly_cost': fixed_yearly_cost_per_point
    })

    # Calculate the total yearly cost for each service point
    service_point_costs['total_yearly_cost'] = (
        service_point_costs['yearly_delivery_cost'] +
        service_point_costs['yearly_capacity_cost'] +
        service_point_costs['fixed_yearly_cost']
    )

    # Calculate the total demand for each service point
    total_demand_per_point = node_data.groupby('assigned_service_point')['Predicted_Demand'].sum()

    # Calculate the cost of 1 unit of demand for each service point
    service_point_costs['cost_per_unit_demand'] = service_point_costs['total_yearly_cost'] / total_demand_per_point

    # Print out the cost of 1 unit of demand for each service point
    print(service_point_costs[['cost_per_unit_demand']])

    # Calculate the sum of each cost category
    total_yearly_delivery_cost = service_point_costs['yearly_delivery_cost'].sum()
    total_yearly_capacity_cost = service_point_costs['yearly_capacity_cost'].sum()
    total_fixed_yearly_cost = service_point_costs['fixed_yearly_cost'].sum()

    # Print out the total cost for each category
    print(f'Total Yearly Delivery Cost: ${total_yearly_delivery_cost:.2f}')
    print(f'Total Yearly Capacity Cost: ${total_yearly_capacity_cost:.2f}')
    print(f'Total Fixed Yearly Cost: ${total_fixed_yearly_cost:.2f}')

    # Return the total yearly cost for all service points combined
    total_yearly_cost = service_point_costs['total_yearly_cost'].sum()
    return total_yearly_cost

# Example usage
node_demand_file = 'final_data.csv'
node_service_point_file = 'allocations.csv'
capacity_of_point_file = 'capacity_of_point.csv'

total_yearly_cost = calculate_yearly_cost(node_demand_file, node_service_point_file, capacity_of_point_file)

print(f'Total yearly cost: ${total_yearly_cost:.2f}')



#jaki koszt w $ na jeden demand

