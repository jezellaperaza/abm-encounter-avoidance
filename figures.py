from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    x = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 1.5, 3]
    labels = ['Asocial', 'Semi-social', 'Social']

    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()

    for i, weight in enumerate(schooling_weights):
        prob = np.array(probabilities[i])
        prob_normalized = prob / simulation.NUM_FISHES

        mean_prob = np.mean(prob_normalized, axis=1)
        min_prob = np.min(prob_normalized, axis=1)
        max_prob = np.max(prob_normalized, axis=1)

        # Plot mean probability
        plt.errorbar(x, mean_prob, yerr=[mean_prob - min_prob, max_prob - mean_prob], fmt='o', label=labels[i],
                     capsize=5)

    plt.xticks(x, rotation=45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
    plt.xlabel('Tidal Flow (m/s)')
    plt.ylabel('Risk probabilities')
    plt.title(title)
    plt.legend()
    plt.show()

def main():

    num_simulations = 5
    schooling_weights = [0, 0.5, 1]
    flow_speeds = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 1.5, 3]

    all_counts = {
        'fish_in_zoi': [],
        'fish_in_ent': [],
        'fish_collided': [],
        'fish_struck': [],
        'fish_collided_struck': []
    }
    all_probabilities = {
        'zoi': [],
        'ent': [],
        'collide': [],
        'strike': [],
        'collide_struck': []
    }

    for weight in schooling_weights:
        zoi_probabilities = []
        ent_probabilities = []
        collide_probabilities = []
        strike_probabilities = []
        collide_strike_probabilities = []

        for flow_speed in flow_speeds:
            simulation.SCHOOLING_WEIGHT = weight
            simulation.FLOW_SPEED = flow_speed
            zoi_probabilities_flow = []
            ent_probabilities_flow = []
            collide_probabilities_flow = []
            strike_probabilities_flow = []
            collide_strike_probabilities_flow = []

            for sim_num in range(num_simulations):
                world = simulation.World()

                world.update()

                # store simulation results
                all_counts['fish_in_zoi'].append(world.fish_in_zoi_count)
                all_counts['fish_in_ent'].append(world.fish_in_ent_count)
                all_counts['fish_collided'].append(world.fish_collided_count)
                all_counts['fish_struck'].append(world.fish_struck_count)
                all_counts['fish_collided_struck'].append(world.fish_collided_and_struck_count)

                zoi_probabilities_flow.append(world.fish_in_zoi_count)
                ent_probabilities_flow.append(world.fish_in_ent_count)
                collide_probabilities_flow.append(world.fish_collided_count)
                strike_probabilities_flow.append(world.fish_struck_count)
                collide_strike_probabilities_flow.append(world.fish_collided_and_struck_count)

            zoi_probabilities.append(zoi_probabilities_flow)
            ent_probabilities.append(ent_probabilities_flow)
            collide_probabilities.append(collide_probabilities_flow)
            strike_probabilities.append(strike_probabilities_flow)
            collide_strike_probabilities.append(collide_probabilities_flow)

        all_probabilities['zoi'].append(zoi_probabilities)
        all_probabilities['ent'].append(ent_probabilities)
        all_probabilities['collide'].append(collide_probabilities)
        all_probabilities['strike'].append(strike_probabilities)
        all_probabilities['collide_struck'].append(collide_strike_probabilities)

    # Create histograms for each set of data
    # fish_occurrence_histogram(all_counts['fish_in_zoi'], 'Fish Probabilities within the Zone of Influence')
    # fish_occurrence_histogram(all_counts['fish_in_ent'], 'Fish Probabilities within Entrainment')
    # fish_occurrence_histogram(all_counts['fish_collided'], 'Probabilities of Fish that Collided with the Turbine')
    # fish_occurrence_histogram(all_counts['fish_struck'], 'Probabilities of Fish Struck by the Turbine')
    # fish_occurrence_histogram(all_counts['fish_collided_struck'], 'Probabilities of Fish Collide and Struck by the Turbine')

    # Plot mean probabilities
    plot_mean_probability(all_probabilities['zoi'], schooling_weights, 'Mean probability of fish being within the zone of influence')
    plot_mean_probability(all_probabilities['ent'], schooling_weights, 'Mean probability of fish being within entrainment')
    plot_mean_probability(all_probabilities['collide'], schooling_weights, 'Mean probability of fish colliding with the turbine')
    plot_mean_probability(all_probabilities['strike'], schooling_weights, 'Mean probability of fish being struck by the turbine')
    plot_mean_probability(all_probabilities['collide_struck'], schooling_weights, 'Mean probability of fish colliding and being struck by the turbine')

main()