from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import math


class World():
    """contains references to all the important stuff in the simulation"""

    NUM_FISHES = 100
    SIZE = (600, 200, 55)
    # Specifies the number of dimensions in the simulation
    # If 2, then the dimensions are [X, Y]
    # If 3, then the dimensions are [X, Y, Z]
    DIMENSIONS = 3
    TURBINE_RADIUS = 5
    TURBINE_POSITION = (SIZE[0] - 25, SIZE[1] / 2, 0)
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
    k = -0.1
    repulsion_strength_at_zero = 1
    avoidance = repulsion_strength_at_zero * math.exp(k * distance)
    return max(0.0, avoidance)


def desired_new_heading(fish: Fish, world: World):
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
    avoidance_direction = np.zeros(World.DIMENSIONS)
    avoidance_found = False

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
                if fish.BLADE_STRIKE_PROBABILITY[0] <= random_probability_of_strike <= fish.BLADE_STRIKE_PROBABILITY[
                    -1]:
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
    attraction_orientation_found = False
    attraction_orientation_direction = np.zeros(World.DIMENSIONS)
    for other, distance in others:
        if distance <= Fish.ATTRACTION_DISTANCE:
            attraction_orientation_found = True
            new_direction = (other.position - fish.position)
            attraction_orientation_direction += (
                    Fish.ATTRACTION_ALIGNMENT_WEIGHT * new_direction / np.linalg.norm(new_direction))

        if distance <= Fish.ORIENTATION_DISTANCE:
            attraction_orientation_found = True
            attraction_orientation_direction += (1 - Fish.ATTRACTION_ALIGNMENT_WEIGHT) * other.heading

        # informed direction makes all fish go a specific direction,
        # with an added weight between preferred direction and social behaviors
        # 0 is all social, and 1 is all preferred direction
        desired_direction = np.zeros(World.DIMENSIONS)
        desired_direction[0] = 1
        informed_direction = desired_direction * Fish.DESIRED_DIRECTION_WEIGHT
        social_direction = (1 - Fish.DESIRED_DIRECTION_WEIGHT) * attraction_orientation_direction

        # the sum vector of all vectors
        # informed direction, social direction, and avoidance
        attraction_orientation_direction = (informed_direction + social_direction) * (1 - strength) + (
                strength * avoidance_direction)

    if attraction_orientation_found:
        norm = np.linalg.norm(attraction_orientation_direction)
        if norm != 0.0:
            return attraction_orientation_direction / norm

    return None


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
    ATTRACTION_DISTANCE = 20
    ORIENTATION_DISTANCE = 15
    ATTRACTION_ALIGNMENT_WEIGHT = 0.5
    MAX_TURN = 0.1  # radians
    TURN_NOISE_SCALE = 0.3  # standard deviation in noise
    SPEED = 1
    # DESIRED_DIRECTION = np.array([1, 0])  # Desired direction of informed fish is towards the right when [1, 0]
    # Desired direction is always 1 in the x direction and 0 in all other direction
    DESIRED_DIRECTION_WEIGHT = 0.2  # Weighting term is strength between swimming
    # towards desired direction and schooling (1 is all desired direction, 0 is all
    # schooling and ignoring desired direction
    # FLOW_VECTOR = np.array([1, 0])
    FLOW_SPEED = 3
    REACTION_DISTANCE = 10
    BLADE_STRIKE_PROBABILITY = np.linspace(0.02, 0.13)

    def __init__(self, position, heading):
        """initial values for position and heading"""
        self.position = position
        self.heading = heading
        self.color = 'blue'
        self.all_fish_left = False
        self.left_environment = False
        self.time_in_zoi = 0
        self.time_in_ent = 0

    def move(self):
        self.position += self.heading * Fish.SPEED
        self.position = np.clip(self.position, [0, 0, 0], World.SIZE)

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
        self.position[1] = self.position[1] % World.SIZE[1]

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


def simulate(num_simulations):

    zoi_fish_time_probabilities = []
    ent_fish_time_probabilities = []

    for simulation_num in range(num_simulations):

        world = World()
        fish_collided_with_turbine = set()
        fish_struck_by_turbine = set()

        world.add_turbine(np.array([world.TURBINE_POSITION[0], world.TURBINE_POSITION[1], world.TURBINE_POSITION[2]]),
                          radius=World.TURBINE_RADIUS, turbine_id='Base', color='red')
        world.add_turbine(np.array([world.TURBINE_POSITION[0], world.TURBINE_POSITION[1], world.TURBINE_RADIUS * 2]),
                          radius=World.TURBINE_RADIUS, turbine_id='Blade', color='red')
        world.add_rectangle(World.ENTRAINMENT_POSITION, World.ENTRAINMENT_DIMENSIONS, color='blue')
        world.add_rectangle(World.ZONE_OF_INFLUENCE_POSITION, World.ZONE_OF_INFLUENCE_DIMENSIONS, color='lightcoral')

        for f in range(World.NUM_FISHES):
            initial_position = np.random.rand(World.DIMENSIONS) * World.SIZE
            initial_position[0] = np.random.uniform(0, 100)
            initial_position[2] = min(initial_position[2], World.SIZE[2])
            world.fishes.append(Fish(initial_position, np.random.rand(World.DIMENSIONS)))

        for frame_number in range(10000):
            for f_num, f in enumerate(world.fishes):
                for rectangle in world.rectangles:
                    if rectangle.color == 'lightcoral' and rectangle.position[0] <= f.position[0] <= rectangle.position[0] + \
                            rectangle.dimensions[0] \
                            and rectangle.position[1] <= f.position[1] <= rectangle.position[1] + rectangle.dimensions[1]:
                        f.time_in_zoi += 1

                    if rectangle.color == 'blue' and rectangle.position[0] <= f.position[0] <= rectangle.position[0] + \
                            rectangle.dimensions[0] \
                            and rectangle.position[1] <= f.position[1] <= rectangle.position[1] + rectangle.dimensions[1]:
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


            for f in world.fishes:
                f.update_heading(desired_new_heading(f, world))
            for f in world.fishes:
                f.move()

            world.all_fish_left = all(f.left_environment for f in world.fishes)
            if world.all_fish_left:
                frame_number = frame_number
                print("All fish have left the environment in frame", frame_number)
                break

        for f_num, f in enumerate(world.fishes):
            time_in_zoi_normalized = f.time_in_zoi / frame_number
            time_in_ent_normalized = f.time_in_ent / frame_number
            zoi_fish_time_probabilities.append(time_in_zoi_normalized)
            ent_fish_time_probabilities.append(time_in_ent_normalized)

    return zoi_fish_time_probabilities, ent_fish_time_probabilities

if __name__ == "__main__":
    num_simulations = 5000
    bins = 10
    zoi_fish_time_probabilities, ent_fish_time_probabilities = simulate(num_simulations)

    zoi_filtered_fish_time_counts = [count for count in zoi_fish_time_probabilities if count > 0]
    ent_filtered_fish_time_counts = [count for count in ent_fish_time_probabilities if count > 0]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(zoi_filtered_fish_time_counts, rwidth = 0.8, bins=bins, edgecolor='black', color='cornflowerblue')
    plt.xlabel('Probabilities')
    plt.ylabel('Number of Simulations')
    plt.title('Time Step Probabilities of Fish within the Zone of Influence')
    plt.xlim(0, max(zoi_filtered_fish_time_counts, default=0))

    plt.subplot(1, 2, 2)
    plt.hist(ent_filtered_fish_time_counts, rwidth = 0.8, bins=5, edgecolor='black', color='cornflowerblue')
    plt.xlabel('Probabilities')
    plt.ylabel('Number of Simulations')
    plt.title('Time Step Probabilities of Fish within Entrainment')
    plt.xlim(0, max(ent_filtered_fish_time_counts, default=0))

    plt.savefig('ZOI-ENT-Time-Steps-High-Flow-5000.png')
    plt.show()

