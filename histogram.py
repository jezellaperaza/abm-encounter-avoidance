from __future__ import annotations
import matplotlib.pyplot as plt
from tqdm import tqdm

import simulation

def fish_occurrence_histogram(total_fish_count, title):
    plt.figure(figsize=(8, 8))
    probabilities = [count / simulation.NUM_FISHES for count in total_fish_count if count != 0]
    plt.hist(probabilities, rwidth=0.8, bins=10, edgecolor='black', color='cornflowerblue')
    plt.xlabel('Probability')
    plt.ylabel('Number of Simulations')
    plt.title(title)
    plt.show()

num_simulations = 10
zoi_fish_counts = []
ent_fish_counts = []
collide_fish_counts = []
strike_fish_counts = []
collide_strike_fish_counts = []

for _ in tqdm(range(num_simulations), desc="Simulation progress"):

    world = simulation.World()
    world.run_full_simulation()
    # world.print_close_out_message()

    zoi_fish_counts.append(world.fish_in_zoi_count)
    ent_fish_counts.append(world.fish_in_ent_count)
    collide_fish_counts.append(world.fish_collided_count)
    strike_fish_counts.append(world.fish_struck_count)
    collide_strike_fish_counts.append(world.fish_collided_and_struck_count)

fish_occurrence_histogram(zoi_fish_counts, "Probabilities of being in the Zone of Influence")
fish_occurrence_histogram(ent_fish_counts, "Probabilities of being Entrained")
fish_occurrence_histogram(collide_fish_counts, "Probabilities of Collision")
fish_occurrence_histogram(strike_fish_counts, "Probabilities of Blade Strike")
fish_occurrence_histogram(collide_strike_fish_counts, "Probabilities of Collision and Blade Strike")