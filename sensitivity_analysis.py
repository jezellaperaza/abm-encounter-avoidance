from __future__ import annotations
import os
import csv
from tqdm import tqdm
import simulation
from multiprocessing import Pool, cpu_count

# Output directory for the CSV files
output_dir = '/Users/jezellaperaza/Documents/GitHub/abm-encounter-avoidance/SA-Results/328-Cross-Flow'
os.makedirs(output_dir, exist_ok=True)

simulation.NUM_FISHES = 328
simulation.SCHOOLING_WEIGHT = 0.5
simulation.FLOW_SPEED = 0.5

# Baseline simulation means
baseline_means = {
    "zoi_mean": 0.3663,
    "ent_mean": 0.2588,
    "collision_mean": 0.03987,
    "strike_mean": 0.01380,
    "collision_strike_mean": 0.007265
}

num_simulations = 1000

# Define parameters and values
parameter_settings = [
    {"parameter_name": "MAX_TURN", "values": [0.64, 0.96]},
    {"parameter_name": "TURN_NOISE_SCALE", "values": [0.008, 0.012]},
    {"parameter_name": "TURBINE_EXPONENTIAL_DECAY", "values": [0.08, 0.12]},
    {"parameter_name": "COLLISION_DISTANCE", "values": [0.12, 0.18]},
    {"parameter_name": "ATTRACTION_DISTANCE", "values": [12, 18]},
    {"parameter_name": "ORIENTATION_DISTANCE", "values": [8, 12]},
    {"parameter_name": "INFORMED_DIRECTION_WEIGHT", "values": [0.16, 0.24]},
    {"parameter_name": "ATTRACTION_WEIGHT", "values": [0.16, 0.24]}
]

def run_simulation(parameter_setting):
    parameter_name = parameter_setting["parameter_name"]
    for value in parameter_setting['values']:
        setattr(simulation, parameter_name, value)
        percent_changes_zoi = []
        percent_changes_ent = []
        percent_changes_collision = []
        percent_changes_strike = []
        percent_changes_collide_strike = []
        fish_in_zoi_ratios = []
        fish_in_ent_ratios = []
        fish_collided_ratios = []
        fish_struck_ratios = []
        fish_collided_and_struck_ratios = []

        for sim in tqdm(range(num_simulations), desc=f"Simulating {parameter_name}={value}"):
            world = simulation.World()
            world.run_full_simulation()

            # Calculate percent changes for ZOI, ENT, collision, strike, and collision & strike
            zoi_percent_change = ((world.fish_in_zoi_count / simulation.NUM_FISHES) - baseline_means["zoi_mean"]) / \
                                 baseline_means["zoi_mean"]
            ent_percent_change = ((world.fish_in_ent_count / simulation.NUM_FISHES) - baseline_means["ent_mean"]) / \
                                 baseline_means["ent_mean"]
            collision_percent_change = ((world.fish_collided_count / simulation.NUM_FISHES) - baseline_means[
                "collision_mean"]) / \
                                       baseline_means["collision_mean"]
            strike_percent_change = ((world.fish_struck_count / simulation.NUM_FISHES) - baseline_means[
                "strike_mean"]) / \
                                    baseline_means["strike_mean"]
            collide_strike_percent_change = ((world.fish_collided_and_struck_count / simulation.NUM_FISHES) -
                                             baseline_means["collision_strike_mean"]) / \
                                            baseline_means["collision_strike_mean"]

            fish_in_zoi_ratio = world.fish_in_zoi_count / simulation.NUM_FISHES
            fish_in_ent_ratio = world.fish_in_ent_count / simulation.NUM_FISHES
            fish_collided_ratio = world.fish_collided_count / simulation.NUM_FISHES
            fish_struck_ratio = world.fish_struck_count / simulation.NUM_FISHES
            fish_collided_and_struck_ratio = world.fish_collided_and_struck_count / simulation.NUM_FISHES

            percent_changes_zoi.append(zoi_percent_change)
            percent_changes_ent.append(ent_percent_change)
            percent_changes_collision.append(collision_percent_change)
            percent_changes_strike.append(strike_percent_change)
            percent_changes_collide_strike.append(collide_strike_percent_change)
            fish_in_zoi_ratios.append(fish_in_zoi_ratio)
            fish_in_ent_ratios.append(fish_in_ent_ratio)
            fish_collided_ratios.append(fish_collided_ratio)
            fish_struck_ratios.append(fish_struck_ratio)
            fish_collided_and_struck_ratios.append(fish_collided_and_struck_ratio)

        # File naming and writing
        file_name = f"{parameter_name}_{value}.csv"
        file_path = os.path.join(output_dir, file_name)

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ['Simulation Number', 'ZOI Percent Change', 'ENT Percent Change', 'Collision Percent Change',
                 'Strike Percent Change', 'Collision & Strike Percent Change', 'Fish in ZOI',
                 'Fish in ENT', 'Fish Collided', 'Fish Struck', 'Fish Collided & Struck'])
            for i in range(num_simulations):
                writer.writerow([
                    i + 1,
                    percent_changes_zoi[i],
                    percent_changes_ent[i],
                    percent_changes_collision[i],
                    percent_changes_strike[i],
                    percent_changes_collide_strike[i],
                    fish_in_zoi_ratios[i],
                    fish_in_ent_ratios[i],
                    fish_collided_ratios[i],
                    fish_struck_ratios[i],
                    fish_collided_and_struck_ratios[i],
                ])

# Main function
if __name__ == '__main__':
    num_cores_to_use = cpu_count()
    with Pool(num_cores_to_use) as pool:
        pool.map(run_simulation, parameter_settings)