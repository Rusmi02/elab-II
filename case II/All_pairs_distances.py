import csv
import networkx as nx

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
                "directed": directed.lower() == 'true',  # Convert string to boolean
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

def build_graph(nodes, edges):
    G = nx.DiGraph()
    for node_id in nodes:
        G.add_node(node_id, pos=(nodes[node_id]['x'], nodes[node_id]['y']))
    for edge in edges.values():
        G.add_edge(edge['v1'], edge['v2'], weight=edge['dist'])
        if not edge['directed']:
            G.add_edge(edge['v2'], edge['v1'], weight=edge['dist'])
    return G

def build_undirected_graph(nodes, edges):
    G = nx.Graph()  # Create an undirected graph
    for node_id in nodes:
        G.add_node(node_id, pos=(nodes[node_id]['x'], nodes[node_id]['y']))
    for edge in edges.values():
        G.add_edge(edge['v1'], edge['v2'], weight=edge['dist'])
    return G

def all_pairs_dijkstra(nodes, edges, undirected=True):
    if undirected:
        G = build_undirected_graph(nodes, edges)
    else:
        G = build_graph(nodes, edges)
    all_distances = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    return all_distances

def print_distances_head(distances, num_lines=5):
    node_ids = list(distances.keys())
    print(" ", "\t".join(map(str, node_ids[:num_lines])))
    for i, node_id1 in enumerate(node_ids[:num_lines]):
        print(node_id1, "\t".join(
            f"{distances[node_id1][node_id2]:.2f}" if distances[node_id1][node_id2] != float('inf') else "inf" for
            node_id2 in node_ids[:num_lines]))

def save_distances_to_csv(distances, file_path):
    node_ids = list(distances.keys())
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow([""] + node_ids)
        for node_id1 in node_ids:
            row = [node_id1] + [distances[node_id1][node_id2] if distances[node_id1][node_id2] != float('inf') else 'inf' for node_id2 in node_ids]
            writer.writerow(row)

# Example usage
nodes = read_nodes("nodes.csv")
edges = read_edges("edges.csv")

# For directed graph distances
directed_distances = all_pairs_dijkstra(nodes, edges, undirected=False)

# For undirected graph distances
undirected_distances = all_pairs_dijkstra(nodes, edges, undirected=True)

# Print the head of the directed distances matrix
print("Directed Distances Head:")
print_distances_head(directed_distances)

# Print the head of the undirected distances matrix
print("Undirected Distances Head:")
print_distances_head(undirected_distances)

# Save the directed distances matrix to a CSV file
save_distances_to_csv(directed_distances, "directed_distances.csv")

# Save the undirected distances matrix to a CSV file
save_distances_to_csv(undirected_distances, "undirected_distances.csv")
