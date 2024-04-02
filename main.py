import os
import numpy as np
from tqdm import tqdm
import simulation

# output_dir = '/Users/jezellaperaza/Documents/GitHub/abm-encounter-avoidance/CSV-Results'
output_dir = 'C:/Users/JPeraza/Documents/GitHub/abm-encounter-avoidance/CSV-Results-Fish-164'
os.makedirs(output_dir, exist_ok=True)

# Parameters for the simulation
num_fish_list = [164]
schooling_weights_list = [0, 0.5, 1]
flow_speeds_list = [-0.05, 0, 0.05, 0.1, 0.15, 0.2, 1.5, 3]
num_simulations = 1000

# Define labels for different model components
model_components = ["ZoneOfInfluence", "Entrainment", "Collision", "Strike", "Collision-Strike"]

# Run simulation for each combination of parameters
for num_fish in num_fish_list:
    for schooling_weight in schooling_weights_list:
        for flow_speed in flow_speeds_list:
            # matrices to store results for each model component
            results_matrices = {component: np.zeros((num_simulations, num_fish)) for component in model_components}

            # Run the simulations
            for sim_index in tqdm(range(num_simulations),
                                  desc=f"Simulations for NF={num_fish}, SW={schooling_weight}, FS={flow_speed}"):
                num_fish = num_fish
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
                                    results_matrices[component][
                                        sim_index, fish_index] = fish.fish_in_zoi_frames / total_frames
                            elif component == "Entrainment":
                                if fish.fish_in_ent_frames > 0:
                                    results_matrices[component][
                                        sim_index, fish_index] = fish.fish_in_ent_frames / total_frames
                            elif component == "Collision":
                                if fish.collided_with_turbine > 0:
                                    results_matrices[component][sim_index, fish_index] = 1
                            elif component == "Strike":
                                if fish.struck_by_turbine > 0 or fish.struck_by_turbine > 0:
                                    results_matrices[component][sim_index, fish_index] = 1
                            elif component == "Collision-Strike":
                                if fish.collided_and_struck > 0 or fish.struck_by_turbine > 0:
                                    results_matrices[component][sim_index, fish_index] = 1

                # Save results for each model component after each combination of parameters
                for component in model_components:
                    filename = f"{component}_Fish_{num_fish}_Weight_{schooling_weight}_Flow_Speed_{flow_speed}.csv"
                    filepath = os.path.join(output_dir, filename)
                    if component in ["Collision", "Strike", "Collision-Strike"]:
                        np.savetxt(filepath, np.round(results_matrices[component], decimals=1), fmt='%1.1f',
                                   delimiter=',',
                                   header=','.join(f"Fish {i}" for i in range(1, num_fish + 1)),
                                   comments='')
                    else:
                        np.savetxt(filepath, results_matrices[component], fmt='%1.4f', delimiter=',',
                                   header=','.join(f"Fish {i}" for i in range(1, num_fish + 1)),
                                   comments='')
                # print("Results saved.")