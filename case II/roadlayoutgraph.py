import pandas as pd
import matplotlib.pyplot as plt

# Read nodes and edges datasets with explicit encoding
nodes_df = pd.read_csv("nodes.csv", sep=";", encoding="utf-8")
edges_df = pd.read_csv("edges.csv", sep=";", encoding="utf-8")

# Parse squares to extract X and Y coordinates separately
nodes_df[['Square_X', 'Square_Y']] = nodes_df['SQUARE'].str.extract(r'E(\d+)N(\d+)')
nodes_df[['Square_X', 'Square_Y']] = nodes_df[['Square_X', 'Square_Y']].astype(int)

# Create a dictionary to map node IDs to coordinates
node_coords = dict(zip(nodes_df['NODE ID'], nodes_df[['X', 'Y']].values))

# Plot nodes
plt.scatter(nodes_df['X'], nodes_df['Y'], color='black')

# Plot edges
for index, row in edges_df.iterrows():
    v1 = row['V1']
    v2 = row['V2']
    x1, y1 = node_coords[v1]
    x2, y2 = node_coords[v2]
    plt.plot([x1, x2], [y1, y2], color='blue')

# Set title and labels
plt.title('Layout of Squares with Intersections and Roads')
plt.xlabel('X')
plt.ylabel('Y')

# Show plot
plt.show()

