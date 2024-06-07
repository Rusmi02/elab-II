### Part 1 - distribution of pickup rate - to get cutoff for when to strt using directed distances

import numpy as np
import matplotlib.pyplot as plt


def linear_exponential_pickup_ratio(distance, max_ratio=0.80, decay_distance=2000, exp_factor=0.001):
    """
    Calculate the pickup ratio based on distance from the service point using a linear function
    with a slight exponential adjustment.

    Parameters:
    - distance: Distance from the service point (meters)
    - max_ratio: Maximum pickup ratio at 0 meters (default is 0.80)
    - decay_distance: Distance at which the pickup ratio reaches approximately 0 (default is 2000 meters)
    - exp_factor: Exponential adjustment factor (default is 0.001)

    Returns:
    - ratio: The calculated pickup ratio
    """
    ratio = max_ratio * (1 - distance / decay_distance) * np.exp(-exp_factor * distance)
    return np.clip(ratio, 0, max_ratio)  # Ensure ratio is within [0, max_ratio]


# Distance range from 0 to 2000 meters
x = np.linspace(0, 2000, 500)

# Calculate pickup ratio using the function
pickup_ratio = linear_exponential_pickup_ratio(x)

# Find the distance where the ratio is 0.5
half_ratio_distance = x[np.where(np.isclose(pickup_ratio, 0.5, atol=0.01))[0][0]]

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.plot(x, pickup_ratio, label='Pickup Ratio (Linear + Exponential)')
plt.axvline(x=half_ratio_distance, color='r', linestyle='--',
            label=f'50% Pickup Ratio at {half_ratio_distance:.1f} meters')
plt.text(half_ratio_distance, 0.5, f'{half_ratio_distance:.1f} meters', color='r', verticalalignment='bottom')

plt.xlabel('Distance from Service Point (meters)')
plt.ylabel('Pickup Ratio')
plt.title('Distribution of Pickup Ratio Based on Distance (Linear + Exponential)')
plt.grid(True)
plt.legend()
plt.show()

### Part 2
