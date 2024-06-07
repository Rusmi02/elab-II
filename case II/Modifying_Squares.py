import pandas as pd

# Read the data from the CSV file
df = pd.read_csv('squares.csv', sep=';')


#here just assuming the population is 3 (since less than 5 not reported, we assume a typical dutch household
df['Population'] = df['Population'].fillna(3)


# List of columns to replace NA with 0
columns_to_fill = [
    'Male', 'Female', 'Age0-14', 'Age15-24', 'Age25-44', 'Age45-64', 'Age65+', 'Households',
    'Single-person households', 'Multi-person households w/o kids', 'Single parent households',
    'Two-parent households', 'Houses', 'Home ownership %', 'Rental %', 'Social housing %', 'Vacant houses'
]

# Replace NA values with 0 in the specified columns
df[columns_to_fill] = df[columns_to_fill].fillna(0)

# Ensure certain columns are treated as numeric
numeric_columns = ['Population', 'Male', 'Female', 'Age0-14', 'Age15-24', 'Age25-44', 'Age45-64', 'Age65+', 'Households', 'Houses']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Calculate percentages
df['Male.prop'] = (df['Male'] / df['Population']) * 100
df['Female.prop'] = (df['Female'] / df['Population']) * 100
df[['Age0-14.prop', 'Age15-24.prop', 'Age25-44.prop', 'Age45-64.prop', 'Age65+.prop']] = df[['Age0-14', 'Age15-24', 'Age25-44', 'Age45-64', 'Age65+']].div(df['Population'], axis=0) * 100

# Add new columns expressed as a percentage of the Households column
df[['Single-person households.prop', 'Multi-person households w/o kids.prop', 'Single parent households.prop', 'Two-parent households.prop']] = df[['Single-person households', 'Multi-person households w/o kids', 'Single parent households', 'Two-parent households']].div(df['Households'], axis=0) * 100

# Add ratios
df['hold_house_ratio'] = df['Households'] / df['Houses']
df['pop_house_ratio'] = df['Population'] / df['Houses']
df['pop_hold_ratio'] = df['Population'] / df['Households']

# Calculate income median
def calculate_income_median(income_range):
    if pd.isnull(income_range):  # Handling missing values
        return None
    income_range = income_range.split('-')
    lower_bound = int(income_range[0])
    upper_bound = int(income_range[1].split()[0])  # Remove any additional text after the upper bound
    return (lower_bound + upper_bound) / 2

df['Income_median'] = df['Median household income'].apply(calculate_income_median)

# Display the first row of the DataFrame
print(df.iloc[0].to_string())

# Replace every comma with a dot in the X and Y columns
df['X'] = df['X'].str.replace(',', '.')
df['Y'] = df['Y'].str.replace(',', '.')

# Save the DataFrame to a CSV file
df.to_csv("modified_squares.csv", index=False)


