from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import simulation

# store the means from the baseline simulation
zoi_mean = 0.3
ent_mean = 0.3
collide_mean = 0.02
strike_mean = 0.07

num_simulations = 10
parameter_labels = []

# Define parameters and values
parameter_settings = [
    {"parameter_name": "MAX_TURN", "values": [0.6, 0.4]},
    {"parameter_name": "TURN_NOISE_SCALE", "values": [0.012, 0.008]},
    {"parameter_name": "TURBINE_EXPONENTIAL_DECAY", "values": [-0.12, -0.08]},
    {"parameter_name": "COLLISION_AVOIDANCE_DISTANCE", "values": [1.2, 0.8]},
    {"parameter_name": "ATTRACTION_DISTANCE", "values": [18, 12]},
    {"parameter_name": "ORIENTATION_DISTANCE", "values": [12, 8]},
    {"parameter_name": "INFORMED_DIRECTION_WEIGHT", "values": [0.6, 0.4]},
    {"parameter_name": "ATTRACTION_WEIGHT", "values": [0.6, 0.4]}
]

zoi_percent_change = []
ent_percent_change = []
collide_percent_change = []
strike_percent_change = []

# this is where we change the parameter by +/- 20 %
for parameter_setting in parameter_settings:
    parameter_name = parameter_setting["parameter_name"]
    parameter_values = parameter_setting["values"]

    # Loop through values of the current parameter
    for parameter_value in parameter_values:

        parameter_value_percent_change = []

        # Loop through simulations
        for _ in tqdm(range(num_simulations), desc=f"Simulation progress ({parameter_name} = {parameter_value})"):

            world = simulation.World()
            world.run_full_simulation()

            setattr(simulation, parameter_name, parameter_value)
            # Calculate percent change and append to the list for this parameter value
            parameter_value_percent_change.append((world.fish_in_zoi_count / simulation.NUM_FISHES - zoi_mean) / zoi_mean * 100)
            setattr(simulation, parameter_name, getattr(simulation, f"{parameter_name}"))

        # Append list of percent changes for this parameter value to the main list
        zoi_percent_change.append(parameter_value_percent_change)

## ZONE OF INFLUENCE VIOLIN PLOTS
plt.figure(figsize=(18, 9))

plt.subplot(2, 4, 1)
sns.violinplot(data=zoi_percent_change[:2], palette=["skyblue", "salmon"])
sns.stripplot(data=zoi_percent_change[:2], color="black", jitter=True)
plt.xlabel("Parameter Value")
plt.ylabel("Percent Change")
plt.title("Max Turning Angle")
plt.xticks(ticks=[0, 1], labels=["0.6", "0.4"])

plt.subplot(2, 4, 2)
sns.violinplot(data=zoi_percent_change[2:4], palette=["skyblue", "salmon"])
sns.stripplot(data=zoi_percent_change[2:4], color="black", jitter=True)
plt.xlabel("Parameter Value")
plt.ylabel("Percent Change")
plt.title("Turn Noise Scale")
plt.xticks(ticks=[0, 1], labels=["0.012", "0.008"])

plt.subplot(2, 4, 3)
sns.violinplot(data=zoi_percent_change[4:6], palette=["skyblue", "salmon"])
sns.stripplot(data=zoi_percent_change[4:6], color="black", jitter=True)
plt.xlabel("Parameter Value")
plt.ylabel("Percent Change")
plt.title("Turbine Exponential Decay")
plt.xticks(ticks=[0, 1], labels=["-0.12", "-0.08"])

plt.subplot(2, 4, 4)
sns.violinplot(data=zoi_percent_change[6:8], palette=["skyblue", "salmon"])
sns.stripplot(data=zoi_percent_change[6:8], color="black", jitter=True)
plt.xlabel("Parameter Value")
plt.ylabel("Percent Change")
plt.title("Repulsion Distance")
plt.xticks(ticks=[0, 1], labels=["1.2", "0.8"])

plt.subplot(2, 4, 5)
sns.violinplot(data=zoi_percent_change[8:10], palette=["skyblue", "salmon"])
sns.stripplot(data=zoi_percent_change[8:10], color="black", jitter=True)
plt.xlabel("Parameter Value")
plt.ylabel("Percent Change")
plt.title("Attraction Distance")
plt.xticks(ticks=[0, 1], labels=["18", "12"])

plt.subplot(2, 4, 6)
sns.violinplot(data=zoi_percent_change[10:12], palette=["skyblue", "salmon"])
sns.stripplot(data=zoi_percent_change[10:12], color="black", jitter=True)
plt.xlabel("Parameter Value")
plt.ylabel("Percent Change")
plt.title("Orientation Distance")
plt.xticks(ticks=[0, 1], labels=["12", "8"])

plt.subplot(2, 4, 7)
sns.violinplot(data=zoi_percent_change[12:14], palette=["skyblue", "salmon"])
sns.stripplot(data=zoi_percent_change[12:14], color="black", jitter=True)
plt.xlabel("Parameter Value")
plt.ylabel("Percent Change")
plt.title("Informed Direction Weight")
plt.xticks(ticks=[0, 1], labels=["0.6", "0.4"])

plt.subplot(2, 4, 8)
sns.violinplot(data=zoi_percent_change[14:], palette=["skyblue", "salmon"])
sns.stripplot(data=zoi_percent_change[14:], color="black", jitter=True)
plt.xlabel("Parameter Value")
plt.ylabel("Percent Change")
plt.title("Attraction Weight")
plt.xticks(ticks=[0, 1], labels=["0.6", "0.4"])

plt.tight_layout()
plt.show()