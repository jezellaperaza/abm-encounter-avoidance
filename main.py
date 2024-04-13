import os
import numpy as np
from tqdm import tqdm
import simulation
from multiprocessing import Pool, cpu_count

output_dir = '/Users/jezellaperaza/Documents/GitHub/abm-encounter-avoidance/Axial-Results' # Macbook
# output_dir = 'C:/Users/JPeraza/Documents/GitHub/abm-encounter-avoidance/Results' # SAFS Computer (personal)
# output_dir = 'C:/Users/jezper/PycharmProjects/abm-encounter-avoidance/Results' # Super Computers (313)
os.makedirs(output_dir, exist_ok=True)

# Parameters for the simulation
num_fish_list = [164]
schooling_weights_list = [0, 0.5, 1]
flow_speeds_list = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
num_simulations = 250

# Define labels for different model components
model_components = ["ZoneOfInfluence", "Entrainment", "Collision", "Strike", "Strike-Time-Steps", "Collision-Strike"]


# Define a helper function to run the simulation
def run_simulation(params):
    # Unpack the parameters tuple
    num_fish, schooling_weight, flow_speed = params

    # Create matrices to store results for each model component
    results_matrices = {component: np.zeros((num_simulations, num_fish)) for component in model_components}

    # Run simulations for the specified parameters
    for sim_index in tqdm(range(num_simulations),
                          desc=f"Simulations for NF={num_fish}, SW={schooling_weight}, FS={flow_speed}"):
        # Configure simulation settings
        simulation.NUM_FISHES = num_fish
        simulation.FLOW_SPEED = flow_speed
        simulation.SCHOOLING_WEIGHT = schooling_weight

        # Initialize the world and the first fish
        world = simulation.World()
        fish = simulation.Fish(position=np.zeros(simulation.DIMENSIONS),
                               heading=np.random.rand(simulation.DIMENSIONS) * 2 - 1,
                               fish_id=0,
                               world=world)

        world.run_full_simulation()
        total_frames = world.frame_number * simulation.UPDATE_GRANULARITY

        # Track fish counts and time steps for each model component
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
                    elif component == "Strike-Time-Steps":
                        if fish.fish_in_blade_frames > 0:
                            results_matrices[component][
                                sim_index, fish_index] = fish.fish_in_blade_frames / total_frames
                    elif component == "Collision":
                        if fish.collided_with_turbine_base > 0:
                            results_matrices[component][sim_index, fish_index] = 1
                    elif component == "Strike":
                        if fish.struck_by_turbine_blade > 0:
                            results_matrices[component][sim_index, fish_index] = 1
                    elif component == "Collision-Strike":
                        if fish.collided_and_struck > 0:
                            results_matrices[component][sim_index, fish_index] = 1

        # Save results after each combination of parameters
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


# Main function
if __name__ == '__main__':
    # Define parameter combinations
    parameters = [(num_fish, schooling_weight, flow_speed)
                  for num_fish in num_fish_list
                  for schooling_weight in schooling_weights_list
                  for flow_speed in flow_speeds_list]

    # Set the number of cores to use
    num_cores_to_use = 4

    # Create a pool of workers
    with Pool(num_cores_to_use) as p:
        # Execute the simulations in parallel
        p.map(run_simulation, parameters)