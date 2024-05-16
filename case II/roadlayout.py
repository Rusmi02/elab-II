import csv
import random
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import sqrt

nodes = {}
with open("case II/nodes.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    next(reader)
    for row in reader:
        node_id, x, y, square = row
        nodes[int(node_id)] = (float(x), float(y), str(square))

edges = {}
with open("case II/edges.csv", newline="") as csvfile:
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
        edges[edge_id] = (
            int(v1),
            int(v2),
            float(dist.replace(",", ".")),
            bool(directed),
            str(road_type),
            str(name),
            int(max_speed),
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            str(square_1),
            str(square_2),
            str(square_mid),
        )

service_points = {}
with open("case II/Service_Point_Locations.csv", newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=";")
    next(reader)
    for row in reader:
        sp_id, x, y, square, population, total_deliveries, total_pickups = row
        service_points[int(sp_id)] = (
            float(x),
            float(y),
            str(square),
            int(population),
            int(total_deliveries),
            int(total_pickups),
        )

G = nx.Graph()

for node_id, (x, y, square) in nodes.items():
    G.add_node(node_id, pos=(x, y))

for sp_id, (
    x,
    y,
    square,
    population,
    total_deliveries,
    total_pickups,
) in service_points.items():
    G.add_node(sp_id, pos=(x, y))

for edge_id, (
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
) in edges.items():
    G.add_edge(v1, v2, weight=dist)


def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


for node_id, (x, y, square) in nodes.items():
    nearest_sp = None
    min_distance = float("inf")
    for sp_id, (sp_x, sp_y, _, _, _, _) in service_points.items():
        dist = distance((x, y), (sp_x, sp_y))
        if dist < min_distance:
            min_distance = dist
            nearest_sp = sp_id
    G.nodes[node_id]["nearest_service_point"] = nearest_sp

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

plt.figure(figsize=(10, 6))
node_sizes = [200 if node_id in service_points else 2 for node_id in G.nodes()]
nx.draw(
    G,
    pos=node_positions,
    with_labels=False,
    node_size=node_sizes,
    node_color=node_colors,
    edge_color="gray",
    arrowsize=10,
)
plt.title("Road Network")
plt.show()
