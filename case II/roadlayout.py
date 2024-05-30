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


def build_graph(nodes, edges, service_points):
    G = nx.DiGraph()

    for node_id, data in nodes.items():
        G.add_node(node_id, pos=(data["x"], data["y"]))

    for sp_id, data in service_points.items():
        G.add_node(sp_id, pos=(data["x"], data["y"]))

    for edge_id, data in edges.items():
        if data["directed"] is True:
            G.add_edge(data["v1"], data["v2"], weight=data["dist"])
        else:
            G.add_edge(data["v1"], data["v2"], weight=data["dist"])
            G.add_edge(data["v2"], data["v1"], weight=data["dist"])

    for node_id, node_data in nodes.items():
        shortest_path_lengths = nx.single_source_dijkstra_path_length(
            G, node_id, weight="weight"
        )
        nearest_sp = min(
            service_points.keys(),
            key=lambda sp_id: shortest_path_lengths.get(sp_id, float("inf")),
        )
        G.nodes[node_id]["nearest_service_point"] = nearest_sp

    return G


def calculate_distance_to_nearest_service_point(G, node_id):
    shortest_path_lengths = nx.single_source_dijkstra_path_length(
        G, node_id, weight="weight"
    )
    nearest_sp = G.nodes[node_id]["nearest_service_point"]
    distance_to_sp = shortest_path_lengths.get(nearest_sp, float("inf"))
    return distance_to_sp


def plot_base_graph(G, nodes, edges, service_points):
    node_positions = nx.get_node_attributes(G, "pos")

    all_colors = list(mcolors.CSS4_COLORS.values())
    random.shuffle(all_colors)
    service_point_colors = all_colors[: len(service_points)]

    service_point_color_map = {}
    for i, sp_id in enumerate(service_points.keys()):
        service_point_color_map[sp_id] = service_point_colors[i]

    node_colors = []
    for node_id in G.nodes():
        if node_id in service_points:
            node_colors.append(service_point_color_map[node_id])
        else:
            nearest_sp = G.nodes[node_id].get("nearest_service_point")
            node_colors.append(service_point_color_map.get(nearest_sp, "skyblue"))

    plt.figure(figsize=(12, 8))
    node_sizes = [100 if node_id in service_points else 2 for node_id in G.nodes()]
    nx.draw(
        G,
        pos=node_positions,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color="gray",
        arrows=False,
    )


def plot_squares(squares, parameter):
    max_value = max(
        square_data[parameter]
        for square_data in squares.values()
        if square_data[parameter] != "NA"
    )
    min_value = min(
        square_data[parameter]
        for square_data in squares.values()
        if square_data[parameter] != "NA"
    )

    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    cmap = plt.cm.viridis

    square_patches = []
    for square_id, square_data in squares.items():
        x = square_data["x"]
        y = square_data["y"]
        value = square_data[parameter]
        if value == "NA":
            color = "lightgray"
        else:
            color = cmap(norm(float(value)))
        square_patches.append(
            patches.Rectangle(
                (x, y),
                7000,
                4500,
                linewidth=1,
                edgecolor=color,
                facecolor=color,
                alpha=0.5,
            )
        )

    ax = plt.gca()
    for patch in square_patches:
        ax.add_patch(patch)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label=parameter, ax=ax)

    plt.title("Road Network with Squares Colored by Parameter")
    plt.show()


def main():
    nodes = read_nodes("case II/nodes.csv")
    edges = read_edges("case II/edges.csv")
    service_points = read_service_points("case II/service_points.csv")
    squares = read_squares("case II/squares.csv")

    G = build_graph(nodes, edges, service_points)
    plot_base_graph(G, nodes, edges, service_points)
    plot_squares(squares, "population")

    node_id = 5515
    distance_to_sp = calculate_distance_to_nearest_service_point(G, node_id)
    print(
        f"Distance from node {node_id} to its nearest service point: {distance_to_sp:.2f}m"
    )


if __name__ == "__main__":
    main()
