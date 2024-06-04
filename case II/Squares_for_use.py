import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('squares.csv', sep=';')

#here just assuming the population is 2 (since less than 5 not reported, we assume close to an average dutch household)
df['Population'] = df['Population'].fillna(2).astype(int)

# Save the DataFrame to a CSV file
df.to_csv("usable_squares.csv", index=False)