from __future__ import annotations
import matplotlib.pyplot as plt

import simulation

def fish_occurrence_histogram(data, title):
    plt.figure(figsize=(8, 8))
    non_zero_counts = [count for count in data if count != 0]
    probabilities = [count / 100 for count in non_zero_counts]
    plt.hist(probabilities, rwidth=0.8, bins=5, edgecolor='black', color='cornflowerblue')
    plt.xlabel('Probabilities')
    plt.ylabel('Number of Simulations')
    plt.title(title)
    plt.show()


def main():

    num_simulations = 5

    all_fish_in_zoi_counts = []
    all_fish_in_ent_counts = []
    all_fish_collided_counts = []
    all_fish_struck_counts = []

    for sim_num in range(num_simulations):
        world = simulation.World()

        # update the world to let the fishes interact
        world.update()

        # store all fish
        all_fish_in_zoi_counts.append(world.fish_in_zoi_count)
        all_fish_in_ent_counts.append(world.fish_in_ent_count)
        all_fish_collided_counts.append(world.fish_collided_count)
        all_fish_struck_counts.append(world.fish_struck_count)

    # Create histograms for each set of data
    fish_occurrence_histogram(all_fish_in_zoi_counts, 'Number of Fish within the Zone of Influence')
    fish_occurrence_histogram(all_fish_in_ent_counts, 'Number of Fish within Entrainment')
    fish_occurrence_histogram(all_fish_collided_counts, 'Number of Fish Collided with the Turbine')
    fish_occurrence_histogram(all_fish_struck_counts, 'Number of Fish Struck by the Turbine')

    # Print closing message after all simulations
    world.print_close_out_message()

main()


