from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm
import seaborn as sns

import simulation

# values we are interested in looking at
schooling_weights = [0, 0.5, 1]
flow_speeds = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 1.5, 3]

num_simulations = 2
zoi_fish_counts = []
ent_fish_counts = []
collide_fish_counts = []
strike_fish_counts = []
collide_strike_fish_counts = []

for _ in tqdm(range(num_simulations), desc="Simulation progress"):

    for weight in schooling_weights:
        for flow_speed in flow_speeds:
            simulation.SCHOOLING_WEIGHT = weight
            simulation.FLOW_SPEED = flow_speed
            world = simulation.World()
            world.run_full_simulation()

            zoi_fish_counts.append((flow_speed, weight, world.fish_in_zoi_count))
            print(f"Flow Speed: {flow_speed}, Schooling Weight: {weight}")


def fish_occurrence_scatter(fish_counts, title):
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()

    # Original x values
    labels = ['Asocial', 'Semi-social', 'Social']
    x = np.arange(len(flow_speeds))

    # Define colors for the rectangles
    colors = ['blue', 'orange', 'green']

    for idx, weight in enumerate(schooling_weights):
        probabilities = []
        for speed in flow_speeds:
            # probabilities for each flow speed at a schooling weight
            prob_speed = [count / simulation.NUM_FISHES for (s, w, count) in fish_counts if s == speed and w == weight]
            if prob_speed:
                probabilities.append(prob_speed)

        mean_prob = np.mean(probabilities, axis=1)
        min_prob = np.min(probabilities, axis=1)
        max_prob = np.max(probabilities, axis=1)

        # Calculate the width of each bar
        bar_width = 0.2
        spacing = 0.05

        # Plot transparent rectangles
        for i, (mean_val, min_val, max_val) in enumerate(zip(mean_prob, min_prob, max_prob)):
            ax.add_patch(plt.Rectangle((x[i] + idx * (bar_width + spacing) - bar_width / 2, min_val), bar_width,
                                       max_val - min_val,
                                       color=colors[idx], alpha=0.3))

        # Plot mean values
        ax.plot(x + idx * (bar_width + spacing), mean_prob, marker='o', color=colors[idx], label=labels[idx])

    ax.set_xlabel('Tidal flow (m/s)')
    ax.set_ylabel('Interaction probabilities')
    ax.set_title(title)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(flow_speeds)
    ax.legend()
    plt.show()


fish_occurrence_scatter(zoi_fish_counts, "Probabilities")


def fish_occurrence_whiskers(fish_counts, title):
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()

    # for labeling the plot
    x = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 1.5, 3]
    labels = ['Asocial', 'Semi-social', 'Social']

    for idx, weight in enumerate(schooling_weights):
        probabilities = []
        for speed in flow_speeds:
            # probabilities for each flow speed at a schooling weight
            prob_speed = [count / simulation.NUM_FISHES for (s, w, count) in fish_counts if s == speed and w == weight]
            if prob_speed:
                probabilities.append(prob_speed)

        mean_prob = np.mean(probabilities, axis=1)
        min_prob = np.min(probabilities, axis=1)
        max_prob = np.max(probabilities, axis=1)

        plt.errorbar(x, mean_prob,
                     yerr=[mean_prob - min_prob, max_prob - mean_prob],
                     fmt='o', label=labels[idx], capsize=5)

    plt.xticks(x, rotation=45)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=0.2))
    plt.xlabel('Tidal flow (m/s)')
    plt.ylabel('Interaction probabilities')
    plt.title(title)
    plt.legend()
    plt.show()


fish_occurrence_whiskers(zoi_fish_counts, "Probabilities")


def fish_occurrence_heatmap(fish_counts, title):
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.autolayout"] = True

    labels = ['Asocial', 'Semi-social', 'Social']
    x = [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 1.5, 3]

    # Convert fish counts to a 2D array
    probabilities = np.zeros((len(flow_speeds), len(schooling_weights)))
    for i, speed in enumerate(flow_speeds):
        for j, weight in enumerate(schooling_weights):
            prob_speed = [count / simulation.NUM_FISHES for (s, w, count) in fish_counts if s == speed and w == weight]
            if prob_speed:
                probabilities[i, j] = np.mean(prob_speed)

    # Create heatmap
    sns.heatmap(probabilities, annot=True, cmap='YlGnBu', xticklabels=labels, yticklabels=x)
    plt.xlabel('Schooling')
    plt.ylabel('Tidal flow (m/s)')
    plt.title(title)
    plt.show()


fish_occurrence_heatmap(zoi_fish_counts, "Probabilities")
