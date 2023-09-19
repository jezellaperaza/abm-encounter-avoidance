from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class World():
    """contains references to all the important stuff in the simulation"""

    NUM_FISHES = 64
    SIZE = 100

    def __init__(self):
        self.fishes: list[Fish] = []
        self.turbines: list[Turbine] = []

    def add_turbine(self, position, color='red'):
        turbine = Turbine(position, color)
        self.turbines.append(turbine)

    def initialize_fishes(self):
        for _ in range(self.NUM_FISHES):
            initial_position = np.array([np.random.uniform(0, 10), np.random.rand() * self.SIZE, np.random.rand() * self.SIZE])
            initial_heading = np.random.rand(3) # generates three random numbers between 0 and 1, which represent the x, y, z of the fish's initial heading?
            informed = np.random.choice([True, False])
            self.fishes.append(Fish(initial_position, initial_heading, informed))

class Turbine:
    def __init__(self, points, color='red'):
        self.points = points
        self.color = color

def distance_between(fishA: Fish, fishB: Fish) -> float:
    return np.linalg.norm(fishA.position - fishB.position)

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
    TURN_NOISE_SCALE = 0.1  # standard deviation in noise
    SPEED = 1.0
    DESIRED_DIRECTION = np.array([1, 0, 0])  # Desired direction of informed fish is towards the right when [1, 0, 0]
    DESIRED_DIRECTION_WEIGHT = 0.5  # Weighting term is strength between swimming
    # towards desired direction and schooling (1 is all desired direction, 0 is all
    # schooling and ignoring desired direction
    FLOW_VECTOR = np.array([1, 0, 0])
    FLOW_SPEED = 0.1

    def __init__(self, position, heading, informed=False):
        """initial values for position and heading
        setting up the informed fish from a subset of NUM_FISHES
        pink fish are informed, but fish are uninformed"""
        self.position = position
        self.heading = heading / np.linalg.norm(heading)  # Ensure heading is a unit vector
        self.informed = informed
        self.color = 'pink' if informed else 'blue'
        self.all_fish_left = False
        self.left_environment = False

    def move(self):
        self.position += self.heading * Fish.SPEED

        # Applies circular boundary conditions without worrying about
        # heading decisions.
        self.position = np.mod(self.position, World.SIZE)

    def update_heading(self, new_heading):
        """Assumes self.heading and new_heading are unit vectors"""

        if new_heading is not None:
            # Adding flow to fish's movement including the speed and direction
            new_heading += Fish.FLOW_VECTOR * Fish.FLOW_SPEED
            noise = np.random.normal(0, Fish.TURN_NOISE_SCALE, 3)  # Adding noise to new_heading
            noisy_new_heading = new_heading + noise  # New_heading is combined with generated noise

            dot = np.dot(noisy_new_heading, self.heading)
            dot = min(1.0, dot)
            dot = max(-1.0, dot)
            angle_between = np.arccos(dot)
            if angle_between > Fish.MAX_TURN:
                noisy_new_heading = rotate_towards(self.heading, noisy_new_heading, Fish.MAX_TURN)

            self.heading = noisy_new_heading

def fish_within(point, vertices):
    x, y, z = point
    n = len(vertices)
    inside = False

    for i in range(n):
        j = (i + 1) % n
        xi, yi, zi = vertices[i]
        xj, yj, zj = vertices[j]

        if yi <= y < yj or yj <= y < yi:
            if x > (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside

    return inside

def desired_new_heading(fish: Fish, world: World):
    # Find all pairwise distances
    others: list[(Fish, float)] = []

    for other in world.fishes:
        if other is not fish:
            others.append((other, distance_between(fish, other)))

    # Compute repulsion
    repulsion_found = False
    repulsion_direction = np.array([0.0, 0.0, 0.0])

    for other, distance in others:
        if distance <= Fish.REPULSION_DISTANCE:
            repulsion_found = True
            vector_difference = other.position - fish.position
            repulsion_direction -= (vector_difference / np.linalg.norm(vector_difference))

    if repulsion_found:
        return repulsion_direction / np.linalg.norm(repulsion_direction)

    # If we didn't find anything within the repulsion distance, then we
    # do attraction distance and orientation distance.
    attraction_orientation_found = False
    attraction_orientation_direction = np.array([0.0, 0.0, 0.0])

    for other, distance in others:
        if distance <= Fish.ATTRACTION_DISTANCE:
            attraction_orientation_found = True
            new_direction = (other.position - fish.position)
            attraction_orientation_direction += (Fish.ATTRACTION_ALIGNMENT_WEIGHT * new_direction / np.linalg.norm(new_direction))

        if distance <= Fish.ORIENTATION_DISTANCE:
            attraction_orientation_found = True
            attraction_orientation_direction += (1 - Fish.ATTRACTION_ALIGNMENT_WEIGHT) * other.heading

    # If fish are informed, an informed direction is calculated by multiplying the direction and weight
    if fish.informed:
        informed_direction = Fish.DESIRED_DIRECTION * Fish.DESIRED_DIRECTION_WEIGHT
        attraction_orientation_direction = informed_direction + (1 - Fish.DESIRED_DIRECTION_WEIGHT) * attraction_orientation_direction

    if attraction_orientation_found:
        norm = np.linalg.norm(attraction_orientation_direction)
        if norm != 0.0:
            return attraction_orientation_direction / norm

    return None

def main():
    # Initialize the world and all the fish
    world = World()
    fish_in_zoi = set()
    fish_in_ent = set()
    frame_number = 0

    # Initialize fishes
    world.initialize_fishes()

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = [], [], []
    sc = ax.scatter(x, y, z, s=5)

    # plot limits
    ax.set_xlim(0, World.SIZE)
    ax.set_ylim(0, World.SIZE)
    ax.set_zlim(0, World.SIZE)

    # view angle
    # ax.view_init(elev=0, azim=180)

    def animate(_):
        nonlocal frame_number

        x = [f.position[0] for f in world.fishes]
        y = [f.position[1] for f in world.fishes]
        z = [f.position[2] for f in world.fishes]
        sc._offsets3d = (x, y, z)

        for f in world.fishes:
            f.update_heading(desired_new_heading(f, world))
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
