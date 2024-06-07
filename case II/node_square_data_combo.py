import csv
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from math import sqrt



def read_nodes(file_path):
    nodes = {}
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        next(reader)
        for row in reader:
            node_id, x, y, square = row
            nodes[int(node_id)] = {"x": float(x), "y": float(y), "square": str(square)}
    return nodes


def read_edges(file_path):
    edges = {}
    with open(file_path, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        next(reader)
        for row in reader:
            (
                edge_id,
                v1,
                v2,
                dist,
                directed,
                road_type,
                name,
                max_speed,
                x1,
                y1,
                x2,
                y2,
                square_1,
                square_2,
                square_mid,
            ) = row
            edges[edge_id] = {
                "v1": int(v1),
                "v2": int(v2),
                "dist": float(dist.replace(",", ".")),
                "directed": bool(directed),
                "road_type": str(road_type),
                "name": str(name),
                "max_speed": int(max_speed),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "square_1": str(square_1),
                "square_2": str(square_2),
                "square_mid": str(square_mid),
            }
    return edges


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


def read_squares(file_path):
    squares = {}
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        next(reader)
        for row in reader:
            (
                square_id,
                population,
                male,
                female,
                age_0_14,
                age_15_24,
                age_25_44,
                age_45_64,
                age_65plus,
                households,
                single_person_households,
                multi_person_households_no_kids,
                single_parent_households,
                two_parent_households,
                houses,
                home_ownership_percent,
                rental_percent,
                social_housing_percent,
                vacant_houses,
                avg_home_value,
                urbanization_index,
                median_household_income,
                percent_low_income_households,
                percent_high_income_households,
                distance_nearest_supermarket_km,
                x,
                y,
            ) = row
            squares[str(square_id)] = {
                "population": int(population) if population else "NA",
                "male": int(male) if male else "NA",
                "female": int(female) if female else "NA",
                "age_0_14": int(age_0_14) if age_0_14 else "NA",
                "age_15_24": int(age_15_24) if age_15_24 else "NA",
                "age_25_44": int(age_25_44) if age_25_44 else "NA",
                "age_45_64": int(age_45_64) if age_45_64 else "NA",
                "age_65plus": int(age_65plus) if age_65plus else "NA",
                "households": int(households) if households else "NA",
                "single_person_households": (
                    int(single_person_households) if single_person_households else "NA"
                ),
                "multi_person_households_no_kids": (
                    int(multi_person_households_no_kids)
                    if multi_person_households_no_kids
                    else "NA"
                ),
                "single_parent_households": (
                    int(single_parent_households) if single_parent_households else "NA"
                ),
                "two_parent_households": (
                    int(two_parent_households) if two_parent_households else "NA"
                ),
                "houses": int(houses) if houses else "NA",
                "home_ownership_percent": (
                    float(home_ownership_percent) if home_ownership_percent else "NA"
                ),
                "rental_percent": float(rental_percent) if rental_percent else "NA",
                "social_housing_percent": (
                    float(social_housing_percent) if social_housing_percent else "NA"
                ),
                "vacant_houses": int(vacant_houses) if vacant_houses else "NA",
                "avg_home_value": float(avg_home_value) if avg_home_value else "NA",
                "urbanization_index": (
                    int(urbanization_index) if urbanization_index else "NA"
                ),
                "median_household_income": (
                    str(median_household_income) if median_household_income else "NA"
                ),
                "percent_low_income_households": (
                    float(percent_low_income_households)
                    if percent_low_income_households
                    else "NA"
                ),
                "percent_high_income_households": (
                    float(percent_high_income_households)
                    if percent_high_income_households
                    else "NA"
                ),
                "distance_nearest_supermarket_km": (
                    float(distance_nearest_supermarket_km.replace(",", "."))
                    if distance_nearest_supermarket_km
                    else "NA"
                ),
                "x": float(x.replace(",", ".")) if x else "NA",
                "y": float(y.replace(",", ".")) if y else "NA",
            }
    return squares



import pandas as pd

def read_usable_squares(file_path):
    squares = {}
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)  # Skip the header row
        for row in reader:
            (
                square_id,
                population,
                male,
                female,
                age_0_14,
                age_15_24,
                age_25_44,
                age_45_64,
                age_65plus,
                households,
                single_person_households,
                multi_person_households_no_kids,
                single_parent_households,
                two_parent_households,
                houses,
                home_ownership_percent,
                rental_percent,
                social_housing_percent,
                vacant_houses,
                avg_home_value,
                urbanization_index,
                median_household_income,
                percent_low_income_households,
                percent_high_income_households,
                distance_nearest_supermarket_km,
                x,
                y,
            ) = row
            squares[str(square_id)] = {
                "population": int(population) if population else None,
                "male": float(male) if male else None,
                "female": float(female) if female else None,
                "age_0_14": float(age_0_14) if age_0_14 else None,
                "age_15_24": float(age_15_24) if age_15_24 else None,
                "age_25_44": float(age_25_44) if age_25_44 else None,
                "age_45_64": float(age_45_64) if age_45_64 else None,
                "age_65plus": float(age_65plus) if age_65plus else None,
                "households": float(households) if households else None,
                "single_person_households": float(single_person_households) if single_person_households else None,
                "multi_person_households_no_kids": float(multi_person_households_no_kids) if multi_person_households_no_kids else None,
                "single_parent_households": float(single_parent_households) if single_parent_households else None,
                "two_parent_households": float(two_parent_households) if two_parent_households else None,
                "houses": float(houses) if houses else None,
                "home_ownership_percent": float(home_ownership_percent) if home_ownership_percent else None,
                "rental_percent": float(rental_percent) if rental_percent else None,
                "social_housing_percent": float(social_housing_percent) if social_housing_percent else None,
                "vacant_houses": float(vacant_houses) if vacant_houses else None,
                "avg_home_value": float(avg_home_value) if avg_home_value else None,
                "urbanization_index": int(float(urbanization_index)) if urbanization_index else None,
                "median_household_income": median_household_income if median_household_income else None,
                "percent_low_income_households": float(percent_low_income_households) if percent_low_income_households else None,
                "percent_high_income_households": float(percent_high_income_households) if percent_high_income_households else None,
                "distance_nearest_supermarket_km": float(distance_nearest_supermarket_km.replace(",", ".")) if distance_nearest_supermarket_km else None,
                "x": float(x.replace(",", ".")) if x else None,
                "y": float(y.replace(",", ".")) if y else None,
            }
    return squares

def is_node_within_square(node_x, node_y, square_data):
    # Define the boundaries of the square with (x, y) as the center
    half_width = 7000 / 2
    half_height = 4500 / 2

    x_min = square_data["x"] - half_width
    y_min = square_data["y"] - half_height
    x_max = square_data["x"] + half_width
    y_max = square_data["y"] + half_height

    # Check if the node is within the boundaries of the square
    return x_min <= node_x <= x_max and y_min <= node_y <= y_max

def assign_population(df):
    # Group by 'square' and count the number of nodes in each square
    node_counts = df.groupby('square').size().reset_index(name='node_count')

    # Merge the node counts back into the original DataFrame
    df = df.merge(node_counts, on='square', how='left')

    # Calculate the population per node for each square
    df['assigned_population'] = df.apply(
        lambda row: row['population'] / row['node_count'] if row['square'] else None,
        axis=1
    )

    # Drop the node_count column as it's no longer needed
    df.drop(columns=['node_count'], inplace=True)

    return df

def read_allocations(file_path):
    allocations = []
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            allocations.append((int(row[0]), int(row[1]), float(row[2])))
    return allocations

def create_node_square_dataframe(nodes, squares, allocations):

    node_square_data = []
    for node_id, service_point_id, distance in allocations:
        node_x = nodes[node_id]["x"]
        node_y = nodes[node_id]["y"]
        found_square = None

        for square_id, square_data in squares.items():
            if is_node_within_square(node_x, node_y, square_data):
                found_square = square_id
                break

        assigned_square = nodes[node_id]["square"]

        node_square_data.append({
            "node_id": node_id,
            "x": node_x,
            "y": node_y,
            "square": found_square,
            "assigned_square": assigned_square,
            "population": squares.get(found_square, {}).get('population', 'NA'),
            "assigned_service_point": service_point_id,
            "distance": distance
        })

    df = pd.DataFrame(node_square_data)
    return df


def main():
    nodes = read_nodes("nodes.csv")
    squares = read_usable_squares("usable_squares.csv")
    allocations = read_allocations("allocations.csv")

    df = create_node_square_dataframe(nodes, squares, allocations)


    # Save the dataframe to a CSV file (optional, for inspection)
    df.to_csv("node_square_data.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    main()



if __name__ == "__main__":
    main()


