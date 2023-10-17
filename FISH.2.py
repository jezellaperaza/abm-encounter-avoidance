from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.patches as patches
import random

class World():
    """contains references to all the important stuff in the simulation"""

    NUM_FISHES = 64
    SIZE = 100

    def __init__(self):
        self.fishes: list[Fish] = []
        self.turbines: list[Turbine] = []

    def add_turbine(self, position, radius, color='red'):
        turbine = Turbine(position, radius, color)
        self.turbines.append(turbine)


class Turbine:
    def __init__(self, position, radius, color='red'):
        self.position = np.array(position)
        self.radius = radius
        self.color = color


def distance_between(fishA: Fish, fishB: Fish) -> float:
    return np.linalg.norm(fishA.position - fishB.position)


def avoidance_strength(distance):
    slope = -0.05
    y_intercept = 1.0
    avoidance = slope * distance + y_intercept
    return max(0.0, avoidance)


def apply_flow(new_heading, flow_vector, flow_speed):
    new_heading += flow_vector * flow_speed
    return new_heading


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
    repulsion_direction = np.array([0.0, 0.0])

    for other, distance in others:
        if distance <= Fish.REPULSION_DISTANCE:
            repulsion_found = True
            vector_difference = other.position - fish.position
            repulsion_direction -= (vector_difference / np.linalg.norm(vector_difference))

    if repulsion_found:
        return repulsion_direction / np.linalg.norm(repulsion_direction)

    # avoidance application
    strength = 0
    avoidance_direction = np.array([0.0, 0.0])
    avoidance_found = False
    for turbine in world.turbines:
        if turbine.color == 'red':
            vector_to_fish = fish.position - turbine.position
            distance_to_turbine = np.linalg.norm(vector_to_fish)
            if distance_to_turbine <= 20.0:
                avoidance_found = True
                strength = avoidance_strength(distance_to_turbine)
                # avoidance_direction += (vector_to_fish / distance_to_turbine) * strength
                avoidance_direction += fish.AVOIDANCE_DIRECTION * strength

    if avoidance_found:
        return avoidance_direction / np.linalg.norm(avoidance_direction)

    # If we didn't find anything within the repulsion distance, then we
    # do attraction distance and orientation distance.
    # It's an unweighted sum of all the unit vectors:
    # + pointing towards other fish inside ATTRACTION_DISTANCE
    # + pointing in the same direction as other fish inside ORIENTATION_DISTANCE

    # original code was an unweighted sum, now included ATTRACTION_ALIGNMENT_WEIGHT
    # 1 being all attraction, 0 being all alignment
    attraction_orientation_found = False
    attraction_orientation_direction = np.array([0.0, 0.0])
    for other, distance in others:
        if distance <= Fish.ATTRACTION_DISTANCE:
            attraction_orientation_found = True
            new_direction = (other.position - fish.position)
            attraction_orientation_direction += (Fish.ATTRACTION_ALIGNMENT_WEIGHT * new_direction / np.linalg.norm(new_direction))

        if distance <= Fish.ORIENTATION_DISTANCE:
            attraction_orientation_found = True
            attraction_orientation_direction += (1 - Fish.ATTRACTION_ALIGNMENT_WEIGHT) * other.heading

        informed_direction = Fish.DESIRED_DIRECTION * Fish.DESIRED_DIRECTION_WEIGHT
        social_direction = (1 - Fish.DESIRED_DIRECTION_WEIGHT) * attraction_orientation_direction

        attraction_orientation_direction = (informed_direction + social_direction) * (1 - strength) + (strength * avoidance_direction)

    if attraction_orientation_found:
        norm = np.linalg.norm(attraction_orientation_direction)
        if norm != 0.0:
            return attraction_orientation_direction / norm

    return None


def rotate_towards(v_from, v_towards, max_angle):
    """
    Rotates v_from towards v_towards

    Assumes the angle between vector and towards is greater than max angle
    Assumes v_from and v_towards are not parallel or anti-parallel
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
    ATTRACTION_DISTANCE = 15
    ORIENTATION_DISTANCE = 10
    REPULSION_DISTANCE_FROM_TURBINE = 20
    ATTRACTION_ALIGNMENT_WEIGHT = 0.5
    MAX_TURN = 0.1
    TURN_NOISE_SCALE = 0.1 # standard deviation in noise
    SPEED = 1.0
    DESIRED_DIRECTION = np.array([1, 0]) # Desired direction of informed fish is towards the right when [1, 0]
    DESIRED_DIRECTION_WEIGHT = 0.5 # Weighting term is strength between swimming
                                    # towards desired direction and schooling (1 is all desired direction, 0 is all
                                    # schooling and ignoring desired ditrection
    FLOW_VECTOR = np.array([1, 0])
    FLOW_SPEED = 0.1

    def __init__(self, position, heading):
        """initial values for position and heading"""
        self.position = position
        self.heading = heading
        self.color = 'blue'
        self.all_fish_left = False
        self.left_environment = False
        self.AVOIDANCE_DIRECTION = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])

    def move(self):
        self.position += self.heading * Fish.SPEED

        # Applies circular boundary conditions without worrying about
        # heading decisions.
        # self.position = np.mod(self.position, World.SIZE)

        # periodic boundaries for only top and bottom?
        self.position[1] = self.position[1] % World.SIZE

        # for checking if all fish left the environment
        if self.position[0] < 0 or self.position[0] > World.SIZE:
            self.left_environment = True

    def update_heading(self, new_heading, flow_vector, flow_speed):
        """Assumes self.heading and new_heading are unit vectors"""

        if new_heading is not None:

            # adding flow to fish's movement including the speed and direction
            # need to figure out how to update fish's collision immediately
            # when flow is in place
            new_heading = apply_flow(new_heading, flow_vector, flow_speed)

            noise = np.random.normal(0, Fish.TURN_NOISE_SCALE, 2) # adding noise to new_heading
            noisy_new_heading = new_heading + noise # new_heading is combined with generated noise

            dot = np.dot(noisy_new_heading, self.heading)
            dot = min(1.0, dot)
            dot = max(-1.0, dot)
            angle_between = np.arccos(dot)
            if angle_between > Fish.MAX_TURN:
                noisy_new_heading = rotate_towards(self.heading, noisy_new_heading, Fish.MAX_TURN)

            self.heading = noisy_new_heading

def main():
    # initialize the world and all the fish
    world = World()
    frame_number = 0

    world.add_turbine([60, 60], radius=5, color='red')

    for f in range(World.NUM_FISHES):
        # world.fishes.append(Fish((np.random.rand(2)) * World.SIZE, np.random.rand(2)))
        initial_position = np.array([np.random.uniform(0, 10), np.random.rand() * World.SIZE])
        world.fishes.append(Fish(initial_position, np.random.rand(2)))

    fig, ax = plt.subplots()
    x, y = [], []
    sc = ax.scatter(x, y, s=5)

    turbine_patches = [
        patches.Circle(world.turbines[0].position, world.turbines[0].radius, edgecolor=world.turbines[0].color,
                       facecolor='none')
    ]

    for patch in turbine_patches:
        ax.add_patch(patch)
    plt.xlim(0, World.SIZE)
    plt.ylim(0, World.SIZE)

    def animate(_):
        nonlocal frame_number

        x = [f.position[0] for f in world.fishes]
        y = [f.position[1] for f in world.fishes]
        sc.set_offsets(np.c_[x, y])

        for f in world.fishes:
            f.update_heading(desired_new_heading(f, world), Fish.FLOW_VECTOR, Fish.FLOW_SPEED)
        for f in world.fishes:
            f.move()

        colors = [f.color for f in world.fishes]
        sc.set_color(colors)

        world.all_fish_left = all(f.left_environment for f in world.fishes)
        if world.all_fish_left:
            print("All fish have left the environment in frame", frame_number)

        if world.all_fish_left:
            ani.event_source.stop()

        frame_number += 1

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=2, interval=100, repeat=True)
    plt.show()

main()