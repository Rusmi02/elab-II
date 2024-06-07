import numpy as np
import pandas as pd

# Load the CSV files
distances = pd.read_csv('directed_distances.csv')
# undirected_distances = pd.read_csv('undirected_distances.csv')



# Iterate over each cell of undirected_distances and update distances if the condition is met
#for i in range(len(undirected_distances)):
    #for j in range(1, len(undirected_distances.columns)):  # Start from 1 to disregard the 'node_id' column
       # if undirected_distances.iloc[i, j] < 292.6:
         #   distances.iloc[i, j] = undirected_distances.iloc[i, j]

# Convert DataFrame to numpy array
distances = distances.to_numpy()
distances = np.delete(distances, 0, axis=1)

print(distances[:5, :5])


# Save the numpy array
np.save('usable_distances.npy', distances)
