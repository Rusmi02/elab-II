import csv


def read_distances(file_path):
    distances = {}
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)[1:]  # Skip the first cell
        for row in reader:
            node_id = int(row[0])
            distances[node_id] = {int(headers[i]): float(row[i + 1]) if row[i + 1] != 'inf' else float('inf') for i in
                                  range(len(headers))}
    return distances

def allocate_nodes_to_squares(directed_csv, service_points_node_ids):
    directed_distances = read_directed_distances(directed_csv)

    allocations = []
    for node in directed_distances:
        min_distance = float('inf')
        assigned_service_point = None
        for sp in service_points_node_ids:
            distance = directed_distances[node][sp]
            if distance < min_distance:
                min_distance = distance
                assigned_service_point = sp
        allocations.append((node, assigned_service_point, min_distance))

    return allocations


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


def allocate_nodes_to_squares(directed_csv, undirected_csv, service_points_file):
    directed_distances = read_distances(directed_csv)
    undirected_distances = read_distances(undirected_csv)
    service_points = read_service_points(service_points_file)

    # Extract service point node IDs as an array
    service_point_node_ids = list(service_points.keys())

    allocations = []
    for node in directed_distances:
        min_distance = float('inf')
        assigned_service_point = None
        for sp in service_point_node_ids:
            distance = directed_distances[node][sp]
            if distance < 292.6:
                distance = undirected_distances[node][sp]
            if distance < min_distance:
                min_distance = distance
                assigned_service_point = sp
        allocations.append((node, assigned_service_point, min_distance))

    return allocations


def save_allocations_to_csv(allocations, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["node_id", "assigned_service_point", "distance"])
        for allocation in allocations:
            writer.writerow(allocation)

# Example usage
# Allocate nodes to service points
allocations = allocate_nodes_to_squares("directed_distances.csv", "undirected_distances.csv", "service_points.csv")

print("Done")

# Save the allocations to a CSV file
# save_allocations_to_csv(allocations, "allocations.csv")
