from __future__ import annotations
import os
import csv
from tqdm import tqdm
import simulation

# Output directory for the CSV files
output_dir = '/Users/jezellaperaza/Documents/GitHub/abm-encounter-avoidance/SA-Results'
os.makedirs(output_dir, exist_ok=True)

# Baseline simulation means
baseline_means = {
    "zoi_mean": 0.3
    # "collide_mean": 0.2,
    # "strike_mean": 0.1,
    # "collide_strike": 0.2
}

num_simulations = 2

# Define parameters and values
parameter_settings = [
    {"parameter_name": "MAX_TURN", "values": [0.6, 0.4]},
    {"parameter_name": "TURN_NOISE_SCALE", "values": [0.012, 0.008]}
    # {"parameter_name": "TURBINE_EXPONENTIAL_DECAY", "values": [-0.12, -0.08]},
    # {"parameter_name": "COLLISION_AVOIDANCE_DISTANCE", "values": [1.2, 0.8]},
    # {"parameter_name": "ATTRACTION_DISTANCE", "values": [18, 12]},
    # {"parameter_name": "ORIENTATION_DISTANCE", "values": [12, 8]},
    # {"parameter_name": "INFORMED_DIRECTION_WEIGHT", "values": [0.6, 0.4]},
    # {"parameter_name": "ATTRACTION_WEIGHT", "values": [0.6, 0.4]}
]

for parameter_setting in parameter_settings:
    parameter_name = parameter_setting["parameter_name"]
    for value in parameter_setting['values']:

        setattr(simulation, parameter_name, value)

        percent_changes = []  # Store percent changes for current parameter and value across simulations

        for sim in tqdm(range(num_simulations), desc=f"Simulating {parameter_name}={value}"):
            world = simulation.World()
            world.run_full_simulation()

            # Calculate percent change
            zoi_percent_change = ((world.fish_in_zoi_count / simulation.NUM_FISHES) - baseline_means["zoi_mean"]) / \
                                 baseline_means["zoi_mean"]
            # ent_percent_change = ((world.fish_in_ent_count / simulation.NUM_FISHES) - baseline_means["ent_mean"]) / \
            #                      baseline_means["ent_mean"]
            percent_changes.append(zoi_percent_change)
            # percent_changes.append(ent_percent_change)



        # File naming and writing
        file_name = f"{parameter_name}_{value}.csv"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Simulation Number', 'Percent Change'])
            for i, change in enumerate(percent_changes, 1):
                writer.writerow([i, change])