from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np

import simulation

def fish_occurrence_histogram(data, title):
    plt.figure(figsize=(8, 8))
    non_zero_counts = [count for count in data if count != 0]
    probabilities = [count / simulation.NUM_FISHES for count in non_zero_counts]
    plt.hist(probabilities, rwidth=0.8, bins=5, edgecolor='black', color='cornflowerblue')
    plt.xlabel('Probabilities')
    plt.ylabel('Number of Simulations')
    plt.title(title)
    plt.show()


def plot_mean_probability(probabilities, schooling_weights, title):
    # x-axis tick marks
    x = [0, 0.2, 1.5, 3]
    labels = ['Asocial', 'Medium', 'Social']

    plt.figure(figsize=(10, 6))

    for i, weight in enumerate(schooling_weights):
        prob = np.array(probabilities[i])

        # Divide probabilities by total number of fish
        prob_normalized = prob / simulation.NUM_FISHES

        # Calculate mean probability and standard deviation
        mean_prob = np.mean(prob_normalized, axis=1)
        std_dev_prob = np.std(prob_normalized, axis=1)

        # adding the error bar
        plt.errorbar(x, mean_prob, yerr=std_dev_prob, fmt='o-', label=labels[i], capsize=5)

    plt.xticks(x)
    plt.xlabel('Tidal Flow (m/s)')
    plt.ylabel('Probabilities of Risk')
    plt.title(title)
    plt.legend()
    plt.show()

def main():

    num_simulations = 25
    schooling_weights = [0, 0.5, 1]
    flow_speeds = [0, 0.2, 1.5, 3]

    # TODO: Need to combine these empty arrays so we don't have so many.

    # for the histograms
    all_fish_in_zoi_counts = []
    all_fish_in_ent_counts = []
    all_fish_collided_counts = []
    all_fish_struck_counts = []

    # for the other plots
    all_zoi_probabilities = []
    all_ent_probabilities = []
    all_collide_probabilities = []
    all_strike_probabilities = []

    for weight in schooling_weights:
        zoi_probabilities = []
        ent_probabilities = []
        collide_probabilities = []
        strike_probabilities = []

        for flow_speed in flow_speeds:
            simulation.SCHOOLING_WEIGHT = weight
            simulation.FLOW_SPEED = flow_speed
            zoi_probabilities_flow = []
            ent_probabilities_flow = []
            collide_probabilities_flow = []
            strike_probabilities_flow = []

            for sim_num in range(num_simulations):
                world = simulation.World()

                world.update()

                # store all fish regarding counts
                all_fish_in_zoi_counts.append(world.fish_in_zoi_count)
                all_fish_in_ent_counts.append(world.fish_in_ent_count)
                all_fish_collided_counts.append(world.fish_collided_count)
                all_fish_struck_counts.append(world.fish_struck_count)

                # store probability of being in ZOI, entrainment, collision, and strike
                zoi_probabilities_flow.append(world.fish_in_zoi_count)
                ent_probabilities_flow.append(world.fish_in_ent_count)
                collide_probabilities_flow.append(world.fish_collided_count)
                strike_probabilities_flow.append(world.fish_struck_count)

            zoi_probabilities.append(zoi_probabilities_flow)
            ent_probabilities.append(ent_probabilities_flow)
            collide_probabilities.append(collide_probabilities_flow)
            strike_probabilities.append(strike_probabilities_flow)

        all_zoi_probabilities.append(zoi_probabilities)
        all_ent_probabilities.append(ent_probabilities)
        all_collide_probabilities.append(collide_probabilities)
        all_strike_probabilities.append(strike_probabilities)

    # Create histograms for each set of data
    fish_occurrence_histogram(all_fish_in_zoi_counts, 'Fish Probabilities within the Zone of Influence')
    fish_occurrence_histogram(all_fish_in_ent_counts, 'Fish Probabilities within Entrainment')
    fish_occurrence_histogram(all_fish_collided_counts, 'Probabilities of Fish that Collided with the Turbine')
    fish_occurrence_histogram(all_fish_struck_counts, 'Probabilities of Fish Struck by the Turbine')

    # Plot mean probabilities
    plot_mean_probability(all_zoi_probabilities, schooling_weights, 'Mean Probability being within the Zone of Influence')
    plot_mean_probability(all_ent_probabilities, schooling_weights, 'Mean Probability being within Entrainment')
    plot_mean_probability(all_collide_probabilities, schooling_weights, 'Mean Probability Colliding with Turbine')
    plot_mean_probability(all_strike_probabilities, schooling_weights, 'Mean Probability being Struck by Turbine')

main()
