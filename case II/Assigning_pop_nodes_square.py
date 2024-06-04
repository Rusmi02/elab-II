import csv
import pandas as pd

def read_service_points(file_path):
    service_points = {}
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        next(reader)
        for row in reader:
            sp_id, x, y, square, population, total_deliveries, total_pickups = row
            service_points[int(sp_id)] = {
                "x": float(x),
                "y": float(y),
                "square": str(square),
                "population": int(population),
                "total_deliveries": int(total_deliveries),
                "total_pickups": int(total_pickups),
            }
    return service_points

def read_nodes(file_path):
    nodes = []
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        next(reader)
        for row in reader:
            node_id, x, y, square = row
            nodes.append({"node_id": int(node_id), "x": float(x.replace(',', '.')), "y": float(y.replace(',', '.')), "square": str(square)})
    return nodes


def assign_population_squares(df):
    # Create a dictionary to store square population
    square_population = {}

    # Iterate over rows to populate the dictionary
    for index, row in df.iterrows():
        square = row['square']
        population = row['population']

        # If square is not NaN, update the dictionary
        if not pd.isnull(square):
            if square not in square_population:
                square_population[square] = population

    # Create a new column for assigned population
    df['assigned_population'] = 0

    # Iterate over rows again to assign population
    for index, row in df.iterrows():
        square = row['square']
        population = row['population']

        # If square is not NaN, evenly distribute population
        if not pd.isnull(square):
            square_nodes = df[df['square'] == square]
            num_nodes = len(square_nodes)
            assigned_population = population / num_nodes
            df.loc[df['square'] == square, 'assigned_population'] = assigned_population

    return df

    return df

### To be modified:
def assigning_pop(df):   ### finds squares with no nodes in them, allocates to nearest node if it still has no population
    # Create a dictionary to store square population
    square_population = {}

    # Iterate over rows to populate the dictionary
    for index, row in df.iterrows():
        square = row['square']
        population = row['population']

        # If square is not NaN, update the dictionary
        if not pd.isnull(square):
            if square not in square_population:
                square_population[square] = population

    # Iterate over rows again to assign population for nodes with NaN assigned_population and square
    for index, row in df.iterrows():
        assigned_population = row['assigned_population']
        assigned_square = row['assigned_square']
        square = row['square']
        population = row['population']

        # If assigned_population and square are NaN
        if pd.isnull(assigned_population) and pd.isnull(square):
            # Check if there are non-NaN assigned_population values for the assigned_square
            if not pd.isnull(assigned_square):
                square_population_assigned = square_population.get(assigned_square)
                if square_population_assigned is not None:
                    # Allocate the population of the found square to the node
                    df.loc[index, 'assigned_population'] = square_population_assigned
    return df


def main():
    nodes = read_nodes("nodes.csv")
    # Convert the nodes data to a DataFrame
    nodes_df = pd.DataFrame(nodes)

    # Read the data from the CSV file
    df = pd.read_csv('node_square_data.csv')

    # Apply the function
    df = assign_population_squares(df)

    # Extract the 'square' column from nodes_df and rename it to 'assigned_square'
    assigned_square = nodes_df[['node_id', 'square']].rename(columns={'square': 'assigned_square'})

    # Merge the assigned_square DataFrame with the main DataFrame based on 'node_id'
    df = df.merge(assigned_square, on='node_id', how='left')

    df['assigned_population'] = df['assigned_population'].replace(0, 0.3)

## Printing and csv saving
    # Save the updated DataFrame to a new CSV file
    df.to_csv('pop_assigned_square-basis.csv', index=False)

    # Display the updated DataFrame
    # print(df)
#####

#################################checker!!!!!###################

    # Check population of square E1790N3150
    population_sum = df.loc[df['square'] == 'E1730N3170', 'assigned_population'].sum()
    print("Total assigned population for square E1730N3170:", population_sum)

    # Calculate total population served by service point with node_id 753
    node_753_population = df[df['nearest_service_point'] == 753]['assigned_population'].sum()
    print("Total population served by service point with node_id 753:", node_753_population)

    total_allocated_population = df['assigned_population'].sum()
    print("Total allocated population:", total_allocated_population)

    # Count the number of rows where 'square' field is empty
    empty_square_count = df['square'].isnull().sum()

    print("Number of rows with empty 'square' field:", empty_square_count)


########################################################################

if __name__ == "__main__":
    main()

