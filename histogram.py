from __future__ import annotations
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ks_2samp
from scipy.stats import cramervonmises

import simulation

def fish_occurrence_histogram(total_fish_count, title):
    plt.figure(figsize=(8, 8))
    probabilities = [count / simulation.NUM_FISHES for count in total_fish_count if count != 0]
    plt.hist(probabilities, rwidth=0.8, bins=10, edgecolor='black', color='cornflowerblue')
    plt.xlabel('Probability')
    plt.ylabel('Number of Simulations')
    plt.title(title)
    plt.show()

num_simulations = 3
zoi_fish_counts = []
ent_fish_counts = []
collide_fish_counts = []
strike_fish_counts = []
collide_strike_fish_counts = []

for _ in tqdm(range(num_simulations), desc="Simulation progress (500 runs)"):

    world = simulation.World()
    world.run_full_simulation()

    zoi_fish_counts.append(world.fish_in_zoi_count)
    ent_fish_counts.append(world.fish_in_ent_count)
    collide_fish_counts.append(world.fish_collided_count)
    strike_fish_counts.append(world.fish_struck_count)
    collide_strike_fish_counts.append(world.fish_collided_and_struck_count)


## DISTRIBUTION TESTING

## 500 simulations
num_simulations_500 = 500
zoi_fish_counts_500 = []
ent_fish_counts_500 = []
collide_fish_counts_500 = []
strike_fish_counts_500 = []
collide_strike_fish_counts_500 = []

for _ in tqdm(range(num_simulations_500), desc="Simulation progress (500 runs)"):

    world = simulation.World()
    world.run_full_simulation()
    # world.print_close_out_message()

    zoi_fish_counts_500.append(world.fish_in_zoi_count)
    ent_fish_counts_500.append(world.fish_in_ent_count)
    collide_fish_counts_500.append(world.fish_collided_count)
    strike_fish_counts_500.append(world.fish_struck_count)
    collide_strike_fish_counts_500.append(world.fish_collided_and_struck_count)

## 1000 simulations
num_simulations_1000 = 1000
zoi_fish_counts_1000 = []
ent_fish_counts_1000 = []
collide_fish_counts_1000 = []
strike_fish_counts_1000 = []
collide_strike_fish_counts_1000 = []

for _ in tqdm(range(num_simulations_1000), desc="Simulation progress (1000 runs)"):

    world = simulation.World()
    world.run_full_simulation()

    zoi_fish_counts_1000.append(world.fish_in_zoi_count)
    ent_fish_counts_1000.append(world.fish_in_ent_count)
    collide_fish_counts_1000.append(world.fish_collided_count)
    strike_fish_counts_1000.append(world.fish_struck_count)
    collide_strike_fish_counts_1000.append(world.fish_collided_and_struck_count)

## Two-sample Kolmogorov-Smirnov
zoi_statistic, zoi_p_value = ks_2samp(zoi_fish_counts_500, zoi_fish_counts_1000)
ent_statistic, ent_p_value = ks_2samp(ent_fish_counts_500, ent_fish_counts_1000)
collide_statistic, collide_p_value = ks_2samp(collide_fish_counts_500, collide_fish_counts_1000)
strike_statistic, strike_p_value = ks_2samp(strike_fish_counts_500, strike_fish_counts_1000)
collide_strike_statistic, collide_strike_p_value = ks_2samp(collide_strike_fish_counts_500, collide_strike_fish_counts_1000)

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

if zoi_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")

if ent_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")

if collide_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")

if strike_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")

if collide_strike_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")

## Cramer-von Mises

zoi_statistic, zoi_p_value = cramervonmises(zoi_fish_counts_500, zoi_fish_counts_1000)
ent_statistic, ent_p_value = cramervonmises(ent_fish_counts_500, ent_fish_counts_1000)
collide_statistic, collide_p_value = cramervonmises(collide_fish_counts_500, collide_fish_counts_1000)
strike_statistic, strike_p_value = cramervonmises(strike_fish_counts_500, strike_fish_counts_1000)
collide_strike_statistic, collide_strike_p_value = cramervonmises(collide_strike_fish_counts_500, collide_strike_fish_counts_1000)

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

if zoi_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")

if ent_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")

if collide_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")

if strike_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")

if collide_strike_p_value > alpha:
    print("Samples come from the same distribution (fail to reject H0)")
else:
    print("Samples do not come from the same distribution (reject H0)")