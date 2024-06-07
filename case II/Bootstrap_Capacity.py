import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load your data and calculate row sums
file_path = 'service_point_parcels_picked_up.csv'
data = pd.read_csv(file_path, delimiter=';')

# Drop the first column (assumed to be the day number)
data = data.drop(columns=data.columns[0])

# Remove rows where all values are zero
data = data[(data.T != 0).any()]

# Step 1: Load your data and calculate row sums
file_path = 'service_point_parcels_picked_up.csv'
data = pd.read_csv(file_path, delimiter=';')

# Drop the first column (assumed to be the day number)
data = data.drop(columns=data.columns[0])

# Remove rows where all values are zero
data = data[(data.T != 0).any()]

# Step 2: Calculate mean for each column
column_means = data.mean()

# Step 3: Calculate standard deviation for each column
column_stds = data.std()

# Step 4: Calculate standard deviation as percentage of the mean for each column
std_as_percentage_of_mean = (column_stds / column_means) * 100

# Step 5: Find the maximum of those percentages
max_percentage = std_as_percentage_of_mean.max()

print("Mean for each column:")
print(column_means)
print("\nStandard deviation for each column:")
print(column_stds)
print("\nStandard deviation as percentage of the mean for each column:")
print(std_as_percentage_of_mean)
print("\nMaximum of those percentages:", max_percentage)
