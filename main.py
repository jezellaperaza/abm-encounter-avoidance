import os
import numpy as np
from tqdm import tqdm
import simulation

output_dir = '/Users/jezellaperaza/Documents/GitHub/abm-encounter-avoidance/Results'
os.makedirs(output_dir, exist_ok=True)

# Parameters for the simulation
schooling_weights_list = [0, 0.5, 1]
flow_speeds_list = [0.2, 1.5, 3]
num_simulations = 2

# Define labels for different model components
model_components = ["ZoneOfInfluence", "Entrainment", "Collision", "Strike", "Collision-Strike"]

# Run simulation for each combination of parameters
for schooling_weight in schooling_weights_list:
    for flow_speed in flow_speeds_list:
        # Initialize matrices to store results for each model component
        num_fish = simulation.NUM_FISHES
        results_matrices = {component: np.zeros((num_simulations, num_fish)) for component in model_components}

        # Run the simulations
        for sim_index in tqdm(range(num_simulations), desc=f"Simulations for SW={schooling_weight}, FS={flow_speed}"):
            simulation.FLOW_SPEED = flow_speed
            simulation.SCHOOLING_WEIGHT = schooling_weight

            world = simulation.World()
            fish = simulation.Fish(position=np.zeros(simulation.DIMENSIONS),
                                   heading=np.random.rand(simulation.DIMENSIONS) * 2 - 1,
                                   fish_id=0,
                                   world=world)

            world.run_full_simulation()
            total_frames = world.frame_number * simulation.UPDATE_GRANULARITY

            # Keeping track of fish counts and time steps for each model component
            for fish_index, fish in enumerate(world.fishes):
                if fish_index < num_fish:
                    for component in model_components:
                        if component == "ZoneOfInfluence":
                            if fish.fish_in_zoi_frames > 0:
                                results_matrices[component][sim_index, fish_index] = fish.fish_in_zoi_frames / total_frames
                        elif component == "Entrainment":
                            if fish.fish_in_ent_frames > 0:
                                results_matrices[component][sim_index, fish_index] = fish.fish_in_ent_frames / total_frames
                        elif component == "Collision":
                            if fish.collided_with_turbine > 0:
                                results_matrices[component][sim_index, fish_index] = 1
                        elif component == "Strike":
                            if fish.struck_by_turbine > 0 or fish.struck_by_turbine > 0:
                                results_matrices[component][sim_index, fish_index] = 1
                        elif component == "Collision-Strike":
                            if fish.collided_and_struck > 0 or fish.struck_by_turbine > 0:
                                results_matrices[component][sim_index, fish_index] = 1

        # Save results matrices for each model component after each combination of parameters
        for component in model_components:
            filename = f"{component}_Weight_{schooling_weight}_Flow_Speed_{flow_speed}.csv"
            filepath = os.path.join(output_dir, filename)
            labeled_results_matrix = np.insert(results_matrices[component], 0, np.arange(1, num_simulations + 1), axis=1)
            np.savetxt(filepath, labeled_results_matrix, fmt='%1.4f', delimiter=',',
                       header=','.join(f"Fish {i}" for i in range(1, num_fish + 1)), comments='')

print("All results matrices saved as CSV files.")