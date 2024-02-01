from __future__ import annotations
import random
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

class World():
    """contains references to all the important stuff in the simulation"""

    NUM_FISHES = 100
    SIZE = (100, 100, 100)
    # Specifies the number of dimensions in the simulation
    # If 2, then the dimensions are [X, Y]
    # If 3, then the dimensions are [X, Y, Z]
    DIMENSIONS = 3
    TURBINE_RADIUS = 5
    TURBINE_POSITION = (175, SIZE[0] / 2, 0)
    ENTRAINMENT_DIMENSIONS = (10, 10, 10)
    ZONE_OF_INFLUENCE_DIMENSIONS = (140, 10, 25)
    ENTRAINMENT_POSITION = np.array([TURBINE_POSITION[0] + TURBINE_RADIUS - 20, TURBINE_POSITION[1] - 5, 0])
    ZONE_OF_INFLUENCE_POSITION = np.array([TURBINE_POSITION[0] + TURBINE_RADIUS - 160, TURBINE_POSITION[1] - 5, 0])

    def __init__(self):
        self.fishes: list[Fish] = []
        self.turbines: list[Turbine] = []
        self.rectangles = []

    def add_turbine(self, position, radius, turbine_id, color='red'):
        turbine = Turbine(position, radius, turbine_id, color)
        self.turbines.append(turbine)

    def add_rectangle(self, position, dimensions, color='None'):
        rectangle = Rectangle(position, dimensions, color)
        self.rectangles.append(rectangle)


class Turbine:
    def __init__(self, position, radius, turbine_id, color='red'):
        self.position = np.array(position)
        self.radius = radius
        self.turbine_id = turbine_id
        self.color = color


class Rectangle:
    def __init__(self, position, dimensions, color='blue'):
        self.position = np.array(position)
        self.dimensions = dimensions
        self.color = color


def distance_between(fishA: Fish, fishB: Fish) -> float:
    return np.linalg.norm(fishA.position - fishB.position)


def avoidance_strength(distance):
    '''avoidance approach function,
    where avoidance strength is stronger close fish
    get to the turbine'''
    # can make A smaller than 1 if you don't want
    # the avoidance strength to be 1
    # A = repulsion_strength_at_zero
    k = -0.05
    repulsion_strength_at_zero = 1
    avoidance = repulsion_strength_at_zero * math.exp(k * distance)
    return max(0.0, avoidance)


def desired_new_heading(fish: Fish, world: World, parameters):

    # find all pairwise distances
    others: list[(Fish, float)] = []

    for other in world.fishes:
        if other is not fish:
            others.append((other, distance_between(fish, other)))

    # Compute repulsion
    # use this to make sure we're not messing with float comparison to decide
    # whether we had something inside the repulsion distance:
    repulsion_found = False
    repulsion_direction = np.zeros(World.DIMENSIONS)
    avoidance_found = False
    avoidance_direction = np.zeros(World.DIMENSIONS)
    attraction_orientation_found = False
    attraction_orientation_direction = np.zeros(World.DIMENSIONS)

    for other, distance in others:
        if distance <= Fish.REPULSION_DISTANCE:
            repulsion_found = True
            vector_difference = other.position - fish.position
            repulsion_direction -= (vector_difference / np.linalg.norm(vector_difference))

    if repulsion_found:
        return repulsion_direction / np.linalg.norm(repulsion_direction)

    # avoidance application
    # strength is from the function of distance from the turbine
    # and strength of repulsion to avoid
    strength = 0

    for turbine in world.turbines:
        if turbine.turbine_id == 'Base':
            vector_to_fish = fish.position - turbine.position
            distance_to_turbine = np.linalg.norm(vector_to_fish)
            if distance_to_turbine <= fish.REACTION_DISTANCE:
                avoidance_found = True
                strength = avoidance_strength(distance_to_turbine)
                avoidance_direction += (vector_to_fish / distance_to_turbine) * strength

            # vector pointing from fish to turbine
            if distance_to_turbine < turbine.radius:
                fish.color = 'green'
                new_heading = fish.position - turbine.position
                new_heading /= np.linalg.norm(new_heading)
                fish.heading = new_heading

        if turbine.turbine_id == 'Blade':
            vector_to_fish = fish.position - turbine.position
            distance_to_turbine = np.linalg.norm(vector_to_fish)
            if distance_to_turbine <= fish.REACTION_DISTANCE:
                avoidance_found = True
                strength = avoidance_strength(distance_to_turbine)
                avoidance_direction += (vector_to_fish / distance_to_turbine) * strength

            if distance_to_turbine < turbine.radius:
                random_probability_of_strike = np.random.rand()
                if fish.BLADE_STRIKE_PROBABILITY[0] <= random_probability_of_strike <= fish.BLADE_STRIKE_PROBABILITY[-1]:
                    fish.color = 'purple'

    if avoidance_found:
        avoidance_direction /= np.linalg.norm(avoidance_direction)

    # If we didn't find anything within the repulsion distance, then we
    # do attraction distance and orientation distance.
    # It's an unweighted sum of all the unit vectors:
    # + pointing towards other fish inside ATTRACTION_DISTANCE
    # + pointing in the same direction as other fish inside ORIENTATION_DISTANCE
    # original code was an unweighted sum, now included ATTRACTION_ALIGNMENT_WEIGHT
    # 1 being all attraction, 0 being all alignment
    for other, distance in others:
        if distance <= Fish.ATTRACTION_DISTANCE:
            attraction_orientation_found = True
            new_direction = (other.position - fish.position)
            attraction_orientation_direction += (
                        Fish.ATTRACTION_ALIGNMENT_WEIGHT * new_direction / np.linalg.norm(new_direction))

        if distance <= Fish.ORIENTATION_DISTANCE:
            attraction_orientation_found = True
            attraction_orientation_direction += (1 - Fish.ATTRACTION_ALIGNMENT_WEIGHT) * other.heading

    attraction_orientation_direction = (1 - Fish.DESIRED_DIRECTION_WEIGHT) * attraction_orientation_direction

    if attraction_orientation_found:
        norm = np.linalg.norm(attraction_orientation_direction)
        if norm != 0.0:
            return attraction_orientation_direction / norm

    # informed direction makes all fish go a specific direction,
    # with an added weight between preferred direction and social behaviors
    # 0 is all social, and 1 is all preferred direction
    desired_direction = np.zeros(World.DIMENSIONS)
    desired_direction[0] = 1
    informed_direction = desired_direction * Fish.DESIRED_DIRECTION_WEIGHT

    norm = np.linalg.norm(informed_direction)
    if norm != 0.0:
        return informed_direction / norm


def rotate_towards(v_from, v_towards, max_angle):
    """
    Rotates v_from towards v_towards

    Assumes the angle between vector and towards is greater than max angle
    Assumes v_from and v_towards are not parallel or antiparallel
    Assumes both vectors are unit length
    """
    # v_prime is perpendicular to v_from, in the plane defined by
    # v_from and v_towards. so v_from and v_prime are perpendicular unit
    # vectors that span this plane, and we can rotate v_from towards v_towards
    # by finding the appropriate coordinate weights
    v_prime = v_towards - v_towards * np.dot(v_from, v_towards)
    v_prime = v_prime / np.linalg.norm(v_prime)

    return v_from * np.cos(max_angle) + v_prime * np.sin(max_angle)


class Fish():
    """main agent of the model"""

    # Constants:
    REPULSION_DISTANCE = 1
    ATTRACTION_DISTANCE = 25
    ORIENTATION_DISTANCE = 15
    ATTRACTION_ALIGNMENT_WEIGHT = 0.5
    MAX_TURN = 0.1
    TURN_NOISE_SCALE = 0.1 # standard deviation in noise
    SPEED = 1
    DESIRED_DIRECTION_WEIGHT = 0.01  # Weighting term is strength between swimming
    # towards desired direction and schooling (1 is all desired direction, 0 is all
    # schooling and ignoring desired direction
    FLOW_SPEED = 3
    REACTION_DISTANCE = 10
    BLADE_STRIKE_PROBABILITY = np.linspace(0.02, 0.13)

    def __init__(self, position, heading, fish_id, parameters):
        """initial values for position and heading"""
        self.position = position
        self.heading = heading
        self.color = 'blue'
        self.id = fish_id
        self.all_fish_left = False
        self.left_environment = False
        self.time_in_zoi = 0
        self.time_in_ent = 0

        # Set Fish parameters from the provided parameters
        self.REPULSION_DISTANCE = parameters['repulsion_distance_value']
        self.ATTRACTION_DISTANCE = parameters['attraction_distance_value']
        self.ORIENTATION_DISTANCE = parameters['orientation_distance_value']
        self.MAX_TURN = parameters['max_turn_value']
        self.TURN_NOISE_SCALE = parameters['turn_noise_value']
        self.FLOW_SPEED = parameters['tidal_flow_value']

    def move(self):
        self.position += self.heading * Fish.SPEED

        # adding flow to fish's position including the speed and direction
        # fish are unaware of flow
        # Flow vector is always 1.0 in the x direction; zero in other directions
        flow_vector = np.zeros(World.DIMENSIONS)
        flow_vector[0] = 1.0
        self.position += self.FLOW_SPEED * flow_vector

        # Applies circular boundary conditions without worrying about
        # heading decisions.
        # self.position = np.mod(self.position, World.SIZE)

        # periodic boundaries for only top and bottom
        self.position[1:] = np.mod(self.position[1:], World.SIZE[1:])

        # for checking if all fish left the environment
        if self.position[0] < 0 or self.position[0] > World.SIZE[0]:
            self.left_environment = True

    def update_heading(self, new_heading):
        """Assumes self.heading and new_heading are unit vectors"""

        if new_heading is not None:
            # generating some random noise to the fish.heading
            noise = np.random.normal(0, Fish.TURN_NOISE_SCALE, len(new_heading))  # adding noise to new_heading
            noisy_new_heading = new_heading + noise  # new_heading is combined with generated noise

            dot = np.dot(noisy_new_heading, self.heading)
            dot = min(1.0, dot)
            dot = max(-1.0, dot)
            angle_between = np.arccos(dot)
            if angle_between > Fish.MAX_TURN:
                noisy_new_heading = rotate_towards(self.heading, noisy_new_heading, Fish.MAX_TURN)

            self.heading = noisy_new_heading

def generate_random_parameters(num_sets):
    parameters_list = []

    for _ in range(num_sets):
        tidal_flow_center = [0, 1.5, 3]
        max_turn_center = 0.1
        turn_noise_center = 0.1
        repulsion_distance_center = [0.5, 1, 1.5]
        attraction_distance_center = [12.5, 25, 37.5]
        orientation_distance_center = [7.5, 15, 22.5]
        variation_percentage = 10

        tidal_flow_value = random.choice(tidal_flow_center)
        max_turn_range = [max_turn_center - max_turn_center * variation_percentage / 100,
                          max_turn_center + max_turn_center * variation_percentage / 100]
        max_turn_value = round(random.uniform(*max_turn_range), 3)

        turn_noise_range = [turn_noise_center - turn_noise_center * variation_percentage / 100,
                            turn_noise_center + turn_noise_center * variation_percentage / 100]
        turn_noise_value = round(random.uniform(*turn_noise_range), 3)

        repulsion_distance_value = round(random.choice(repulsion_distance_center), 3)
        attraction_distance_value = round(random.choice(attraction_distance_center), 3)
        orientation_distance_value = round(random.choice(orientation_distance_center), 3)

        parameters = {
            'tidal_flow_value': tidal_flow_value,
            'max_turn_value': max_turn_value,
            'turn_noise_value': turn_noise_value,
            'repulsion_distance_value': repulsion_distance_value,
            'attraction_distance_value': attraction_distance_value,
            'orientation_distance_value': orientation_distance_value
        }

        parameters_list.append(parameters)

    return parameters_list

def simulate(num_simulations, parameters):
    fish_in_zoi_count = []
    fish_in_ent_count = []
    fish_collided_count = []
    fish_struck_count = []
    fish_collided_and_struck_count = []

    zoi_fish_time_probabilities = []
    ent_fish_time_probabilities = []

    for simulation_num in range(num_simulations):
        world = World()
        fish_in_zoi = set()
        fish_in_ent = set()
        fish_collided_with_turbine = set()
        fish_struck_by_turbine = set()
        fish_collided_and_struck = set()

        world.add_turbine(np.array([world.TURBINE_POSITION[0], world.TURBINE_POSITION[1], world.TURBINE_POSITION[2]]),
                          radius=World.TURBINE_RADIUS, turbine_id='Base', color='red')
        world.add_turbine(
            np.array([world.TURBINE_POSITION[0], world.TURBINE_POSITION[1], world.TURBINE_RADIUS * 2]),
            radius=World.TURBINE_RADIUS, turbine_id='Blade', color='red')
        world.add_rectangle(World.ENTRAINMENT_POSITION, World.ENTRAINMENT_DIMENSIONS, color='blue')
        world.add_rectangle(World.ZONE_OF_INFLUENCE_POSITION, World.ZONE_OF_INFLUENCE_DIMENSIONS, color='lightcoral')

        for f in range(World.NUM_FISHES):
            initial_position = np.random.rand(World.DIMENSIONS) * World.SIZE
            initial_position[0] = np.random.uniform(0, 100)
            initial_position[2] = min(initial_position[2], World.SIZE[2])
            world.fishes.append(
                Fish(initial_position,
                     # draw randomly between -1 and +1
                     np.random.rand(World.DIMENSIONS) * 2 - 1, fish_id=f, parameters=parameters))

        for frame_number in range(10000):
            for f_num, f in enumerate(world.fishes):
                for rectangle in world.rectangles:
                    if rectangle.color == 'lightcoral' and rectangle.position[0] <= f.position[0] <= rectangle.position[
                        0] + rectangle.dimensions[0] \
                            and rectangle.position[1] <= f.position[1] <= rectangle.position[1] + rectangle.dimensions[1]:
                        fish_in_zoi.add(f_num)
                        f.time_in_zoi += 1

                    if rectangle.color == 'blue' and rectangle.position[0] <= f.position[0] <= rectangle.position[
                        0] + rectangle.dimensions[0] \
                            and rectangle.position[1] <= f.position[1] <= rectangle.position[1] + rectangle.dimensions[1]:
                        fish_in_ent.add(f_num)
                        f.time_in_ent += 1

                for turbine in world.turbines:
                    if turbine.turbine_id == 'Base':
                        if distance_between(f, turbine) < turbine.radius:
                            fish_collided_with_turbine.add(f_num)

                    if turbine.turbine_id == 'Blade':
                        distance = distance_between(f, turbine)
                        if distance < turbine.radius:
                            if f.color == 'purple':
                                fish_struck_by_turbine.add(f_num)

            fish_collided_and_struck = fish_collided_with_turbine.intersection(fish_struck_by_turbine)

            fish_in_zoi_count.append(len(fish_in_zoi))
            fish_in_ent_count.append(len(fish_in_ent))
            fish_collided_count.append(len(fish_collided_with_turbine))
            fish_struck_count.append(len(fish_struck_by_turbine))
            fish_collided_and_struck_count.append(len(fish_collided_and_struck))

            for f in world.fishes:
                f.update_heading(desired_new_heading(f, world, parameters))
            for f in world.fishes:
                f.move()

            world.all_fish_left = all(f.left_environment for f in world.fishes)
            if world.all_fish_left:
                # print("All fish have left the environment in frame", frame_number)
                break

        for f_num, f in enumerate(world.fishes):
            time_in_zoi_normalized = f.time_in_zoi / frame_number
            time_in_ent_normalized = f.time_in_ent / frame_number
            zoi_fish_time_probabilities.append(time_in_zoi_normalized)
            ent_fish_time_probabilities.append(time_in_ent_normalized)

    return fish_in_zoi_count, fish_in_ent_count, fish_collided_count, fish_struck_count, fish_collided_and_struck_count, zoi_fish_time_probabilities, ent_fish_time_probabilities

def simulate_with_parameters(parameters_list, num_simulations=1):
    all_results = []

    for parameters in tqdm(parameters_list, desc="Simulating", unit="set", bar_format="{l_bar}{bar}{r_bar}"):
        fish_in_zoi_count, fish_in_ent_count, fish_collided_count, fish_struck_count, fish_collided_and_struck_count, zoi_fish_time_probabilities, ent_fish_time_probabilities = simulate(
            num_simulations, parameters)

        all_results.append({
            'parameters': parameters,
            'fish_in_zoi_count': fish_in_zoi_count,
            'fish_in_ent_count': fish_in_ent_count,
            'fish_collided_count': fish_collided_count,
            'fish_struck_count': fish_struck_count,
            'fish_collided_and_struck_count': fish_collided_and_struck_count,
            'zoi_fish_time_probabilities': zoi_fish_time_probabilities,
            'ent_fish_time_probabilities': ent_fish_time_probabilities
        })

    return all_results


if __name__ == "__main__":
    num_simulations = 1 # number of simulations per set
    num_sets = 1000 # number of random parameters generated

    random_parameters = generate_random_parameters(num_sets)
    simulation_results = simulate_with_parameters(random_parameters, num_simulations)

    for parameters in random_parameters:
        for key, value in parameters.items():
            print(f"{key}: {value}")
        print()

    # Simulation results for each set of parameters
    for result in simulation_results:
        parameters = result['parameters']
        fish_in_zoi_count = result['fish_in_zoi_count']
        fish_in_ent_count = result['fish_in_ent_count']
        fish_collided_count = result['fish_collided_count']
        fish_struck_count = result['fish_struck_count']
        fish_collided_and_struck_count = result['fish_collided_and_struck_count']
        zoi_fish_time_probabilities = result['zoi_fish_time_probabilities']
        ent_fish_time_probabilities = result['ent_fish_time_probabilities']

    zoi_fish_time_probabilities = [count for count in zoi_fish_time_probabilities if count > 0]
    ent_fish_time_probabilities = [count for count in ent_fish_time_probabilities if count > 0]

    # Filter out zero from lists
    fish_in_zoi_count = [count for count in fish_in_zoi_count if count > 0]
    fish_in_ent_count = [count for count in fish_in_ent_count if count > 0]
    fish_collided_count = [count for count in fish_collided_count if count > 0]
    fish_struck_count = [count for count in fish_struck_count if count > 0]
    fish_collided_and_struck_count = [count for count in fish_collided_and_struck_count if count > 0]

    fish_in_zoi_probabilities = [count / World.NUM_FISHES for count in fish_in_zoi_count]
    fish_in_ent_probabilities = [count / World.NUM_FISHES for count in fish_in_ent_count]
    fish_collided_probabilities = [count / World.NUM_FISHES for count in fish_collided_count]
    fish_struck_probabilities = [count / World.NUM_FISHES for count in fish_struck_count]
    fish_collided_and_struck_probabilities = [count / World.NUM_FISHES for count in fish_collided_and_struck_count]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(zoi_fish_time_probabilities, bins=10, edgecolor='black', color='cornflowerblue')
    plt.xlabel('Probabilities')
    plt.ylabel('Number of Simulations')
    plt.title('Time Step Probabilities of Fish within the Zone of Influence')
    plt.xlim(0, max(zoi_fish_time_probabilities, default=0))

    plt.subplot(1, 2, 2)
    plt.hist(ent_fish_time_probabilities, bins=5, edgecolor='black', color='cornflowerblue')
    plt.xlabel('Probabilities')
    plt.ylabel('Number of Simulations')
    plt.title('Time Step Probabilities of Fish within Entrainment')
    plt.xlim(0, max(ent_fish_time_probabilities, default=0))
    plt.show()

    # Plot histograms with mean probability lines
    def plot_histogram(probabilities, title, num_bins=10):
        plt.hist(probabilities, bins=num_bins, edgecolor='black', color='cornflowerblue')
        plt.ylabel("Frequency of Simulations")
        plt.xlabel(f"Probability")
        plt.title(f"Probability of {title}")
        plt.axvline(np.mean(probabilities), color='r', linestyle='dashed', linewidth=2, label='Mean probability')
        plt.legend()
        plt.xlim(0, max(probabilities, default=0) + 0.05)
        plt.show()

    plot_histogram(fish_in_zoi_probabilities, "Fish in the Zone of Influence", num_bins=10)
    plot_histogram(fish_in_ent_probabilities, "Fish Entrained", num_bins=10)
    plot_histogram(fish_collided_probabilities, "Fish Collided with the Turbine", num_bins=5)
    plot_histogram(fish_struck_probabilities, "Fish Struck by the Turbine", num_bins=5)
    plot_histogram(fish_collided_and_struck_probabilities, "Fish Collided and Struck by the Turbine", num_bins=5)