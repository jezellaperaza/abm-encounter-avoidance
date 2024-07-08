from __future__ import annotations
from tqdm import tqdm
from scipy.stats import ks_2samp
import simulation
import multiprocessing

def run_simulation(num_simulations, description):
    zoi_fish_counts = []
    ent_fish_counts = []
    collide_fish_counts = []
    strike_fish_counts = []
    collide_strike_fish_counts = []

    for _ in tqdm(range(num_simulations), desc=description):
        world = simulation.World()
        world.run_full_simulation()

        zoi_fish_counts.append(world.fish_in_zoi_count)
        ent_fish_counts.append(world.fish_in_ent_count)
        collide_fish_counts.append(world.fish_collided_count)
        strike_fish_counts.append(world.fish_struck_count)
        collide_strike_fish_counts.append(world.fish_collided_and_struck_count)

    return zoi_fish_counts, ent_fish_counts, collide_fish_counts, strike_fish_counts, collide_strike_fish_counts

def print_results_and_interpret(zoi_statistic, zoi_p_value, ent_statistic, ent_p_value, collide_statistic, collide_p_value, strike_statistic, strike_p_value, collide_strike_statistic, collide_strike_p_value):
    print("ZOI KS Statistic:", zoi_statistic)
    print("ZOI P-value:", zoi_p_value)

    print("Entrainment KS Statistic:", ent_statistic)
    print("Entrainment P-value:", ent_p_value)

    print("Collision KS Statistic:", collide_statistic)
    print("Collision P-value:", collide_p_value)

    print("Strike KS Statistic:", strike_statistic)
    print("Strike P-value:", strike_p_value)

    print("Collision, Strike KS Statistic:", collide_strike_statistic)
    print("Collision, Strike P-value:", collide_strike_p_value)

    # Interpret the results
    alpha = 0.05

    print("ZOI:")
    if zoi_p_value > alpha:
        print("Samples come from the same distribution (fail to reject H0)")
    else:
        print("Samples do not come from the same distribution (reject H0)")

    print("Entrainment:")
    if ent_p_value > alpha:
        print("Samples come from the same distribution (fail to reject H0)")
    else:
        print("Samples do not come from the same distribution (reject H0)")

    print("Collision:")
    if collide_p_value > alpha:
        print("Samples come from the same distribution (fail to reject H0)")
    else:
        print("Samples do not come from the same distribution (reject H0)")

    print("Strike:")
    if strike_p_value > alpha:
        print("Samples come from the same distribution (fail to reject H0)")
    else:
        print("Samples do not come from the same distribution (reject H0)")

    print("Collision, Strike:")
    if collide_strike_p_value > alpha:
        print("Samples come from the same distribution (fail to reject H0)")
    else:
        print("Samples do not come from the same distribution (reject H0)")

if __name__ == '__main__':
    num_simulations_500 = 500
    num_simulations_1000 = 1000

    # Define the number of processes
    num_processes = multiprocessing.cpu_count()

    # Divide the simulations equally among processes
    simulations_per_process_500 = num_simulations_500 // num_processes
    simulations_per_process_1000 = num_simulations_1000 // num_processes

    # Create pool of processes
    pool = multiprocessing.Pool(processes=num_processes)

    # Run simulations in parallel
    results_500 = pool.starmap(run_simulation, [(simulations_per_process_500, f'Simulation progress (500 runs)') for _ in range(num_processes)])
    results_1000 = pool.starmap(run_simulation, [(simulations_per_process_1000, f'Simulation progress (1000 runs)') for _ in range(num_processes)])

    pool.close()
    pool.join()

    # Combine results from all processes
    zoi_fish_counts_500 = [item for sublist in results_500 for item in sublist[0]]
    ent_fish_counts_500 = [item for sublist in results_500 for item in sublist[1]]
    collide_fish_counts_500 = [item for sublist in results_500 for item in sublist[2]]
    strike_fish_counts_500 = [item for sublist in results_500 for item in sublist[3]]
    collide_strike_fish_counts_500 = [item for sublist in results_500 for item in sublist[4]]

    zoi_fish_counts_1000 = [item for sublist in results_1000 for item in sublist[0]]
    ent_fish_counts_1000 = [item for sublist in results_1000 for item in sublist[1]]
    collide_fish_counts_1000 = [item for sublist in results_1000 for item in sublist[2]]
    strike_fish_counts_1000 = [item for sublist in results_1000 for item in sublist[3]]
    collide_strike_fish_counts_1000 = [item for sublist in results_1000 for item in sublist[4]]

    # Calculate KS statistics and p-values
    zoi_statistic, zoi_p_value = ks_2samp(zoi_fish_counts_500, zoi_fish_counts_1000)
    ent_statistic, ent_p_value = ks_2samp(ent_fish_counts_500, ent_fish_counts_1000)
    collide_statistic, collide_p_value = ks_2samp(collide_fish_counts_500, collide_fish_counts_1000)
    strike_statistic, strike_p_value = ks_2samp(strike_fish_counts_500, strike_fish_counts_1000)
    collide_strike_statistic, collide_strike_p_value = ks_2samp(collide_strike_fish_counts_500, collide_strike_fish_counts_1000)

    # Print results and interpretations
    print_results_and_interpret(zoi_statistic, zoi_p_value, ent_statistic, ent_p_value, collide_statistic, collide_p_value, strike_statistic, strike_p_value, collide_strike_statistic, collide_strike_p_value)
