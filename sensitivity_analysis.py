from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

import simulation

# store the means from the baseline simulation
zoi_mean = 0.3
ent_mean = 0.3
collide_mean = 0.02
strike_mean = 0.07

num_simulations = 3

# Define parameters to vary and their ranges
parameter_settings = [
    {"parameter_name": "MAX_TURN", "values": np.linspace(0.1, 0.5, num_simulations)},
    {"parameter_name": "TURN_NOISE_SCALE", "values": np.linspace(0.01, 0.1, num_simulations)}
]
parameter_labels = []

zoi_percent_change = []
ent_percent_change = []
collide_percent_change = []
strike_percent_change = []

# this is where we change the parameter by +/- 20 %
for parameter_setting in parameter_settings:
    parameter_name = parameter_setting["parameter_name"]
    parameter_values = parameter_setting["values"]

    # Loop through simulations, varying the current parameter
    for parameter_value in tqdm(parameter_values, desc=f"Simulation progress ({parameter_name})"):
        # Reset simulation world
        world = simulation.World()
        world.run_full_simulation()

        # Set current parameter value
        setattr(simulation, parameter_name, parameter_value)

        # Calculate percent change for ZOI parameter
        zoi_percent_change.append((world.fish_in_zoi_count / simulation.NUM_FISHES - zoi_mean) / zoi_mean * 100)

        # Reset parameter to default after each simulation
        setattr(simulation, parameter_name, getattr(simulation, f"{parameter_name}"))

# Create violin plots
plt.figure(figsize=(10, 6))

# Violin plot for MAX_TURN
plt.subplot(1, 2, 1)
sns.violinplot(data=zoi_percent_change[:num_simulations], color="salmon")
sns.stripplot(data=zoi_percent_change[:num_simulations], color='black', size=5, jitter=True)
plt.xlabel('MAX_TURN')
plt.ylabel('Percent Change of ZOI')
plt.title('Percent Change of ZOI vs. MAX_TURN')
plt.gca().xaxis.set_ticks([])  # Remove x-axis ticks

# Violin plot for TURN_NOISE_SCALE
plt.subplot(1, 2, 2)
sns.violinplot(data=zoi_percent_change[num_simulations:], color="salmon")
sns.stripplot(data=zoi_percent_change[num_simulations:], color='black', size=5, jitter=True)
plt.xlabel('TURN_NOISE_SCALE')
plt.ylabel('Percent Change of ZOI')
plt.title('Percent Change of ZOI vs. TURN_NOISE_SCALE')
plt.gca().xaxis.set_ticks([])  # Remove x-axis ticks

plt.tight_layout()
plt.show()