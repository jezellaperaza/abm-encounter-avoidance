from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import seaborn as sns
import os

import simulation

output_dir = "C:/Users/JPeraza/Documents/GitHub/abm-encounter-avoidance/figures"
os.makedirs(output_dir, exist_ok=True)

# values we are interested in looking at
schooling_weights = [0, 0.5, 1]
flow_speeds = [-0.05, 0, 0.05, 0.1, 0.15, 0.2, 1.5, 3]
num_simulations = 30

zoi_fish_counts = []
ent_fish_counts = []
collide_fish_counts = []
strike_fish_counts = []
collide_strike_fish_counts = []

zoi_fish_time = []

for _ in tqdm(range(num_simulations), desc="Simulation progress"):

    for weight in schooling_weights:
        for flow_speed in flow_speeds:

            simulation.SCHOOLING_WEIGHT = weight
            simulation.FLOW_SPEED = flow_speed

            world = simulation.World()
            fish = simulation.Fish(position=np.zeros(simulation.DIMENSIONS),
                                   heading=np.random.rand(simulation.DIMENSIONS) * 2 - 1,
                                   fish_id=0,
                                   world=world)
            world.run_full_simulation()
            total_frames = world.frame_number * simulation.UPDATE_GRANULARITY

            zoi_fish_counts.append((flow_speed, weight, world.fish_in_zoi_count))

            for fish in world.fishes:
                fish_time_in_zoi = fish.fish_in_zoi_frames / total_frames
                zoi_fish_time.append((flow_speed, weight, fish_time_in_zoi))

            print(f"Flow speed: {flow_speed} Schooling Weight: {weight}")


def fish_occurrence_scatter(fish_counts, title):
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()

    # Original x values
    labels = ['Asocial', 'Semi-social', 'Social']
    x = np.arange(len(flow_speeds))

    # Define colors for the rectangles
    colors = ['blue', 'orange', 'green']

    for idx, weight in enumerate(schooling_weights): # loops over each weight amd indexing
        probabilities = []
        for speed in flow_speeds: # loops over the list of flow speeds
            # probabilities for each flow speed at a schooling weight
            prob_each_speed = [count / simulation.NUM_FISHES for (s, w, count) in fish_counts if s == speed and w == weight] # calculates the prob
            # at each speed for the current weight
            if prob_each_speed: # makes sure probs are not empty then append
                probabilities.append(prob_each_speed)

        if not probabilities:
            continue

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
        ax.plot(x + idx * (bar_width + spacing), mean_prob, marker='o', color=colors[idx], linestyle='', label=labels[idx])


    ax.set_xlabel('Tidal flow (m/s)')
    ax.set_ylabel('Interaction probabilities')
    ax.set_title(title)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(flow_speeds)
    ax.legend()
    plt.savefig(os.path.join(output_dir, title.replace(" ", "_") + "_scatter.png"))
    plt.show()


def fish_time_scatter(fish_counts, title):
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots()

    # Original x values
    labels = ['Asocial', 'Semi-social', 'Social']
    x = np.arange(len(flow_speeds))

    # Define colors for the rectangles
    colors = ['blue', 'orange', 'green']

    for idx, weight in enumerate(schooling_weights):  # loops over each weight amd indexing
        probabilities = []
        for speed in flow_speeds:  # loops over the list of flow speeds
            # probabilities for each flow speed at a schooling weight
            prob_each_speed = [count for (s, w, count) in fish_counts if
                               s == speed and w == weight and count != 0]  # calculates the prob
            # at each speed for the current weight
            if prob_each_speed:  # makes sure probs are not empty then append
                probabilities.append(prob_each_speed)

        if not probabilities:
            continue

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
        ax.plot(x + idx * (bar_width + spacing), mean_prob, marker='o', color=colors[idx], linestyle='',
                label=labels[idx])

    ax.set_xlabel('Tidal flow (m/s)')
    ax.set_ylabel('Interaction probabilities')
    ax.set_title(title)
    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(flow_speeds)
    ax.legend()
    plt.show()


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
    plt.savefig(os.path.join(output_dir, title.replace(" ", "_") + "_heatmap.png"))
    plt.show()