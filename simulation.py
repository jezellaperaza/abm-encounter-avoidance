from __future__ import annotations
import numpy as np
import math

# np.random.seed(123)

## WORLD PARAMETERS
NUM_FISHES = 100
WORLD_SIZE = (400, 100, 55)
DIMENSIONS = len(WORLD_SIZE)
# If this is greater than 1, (say 5), we'll make 5 mini 1/5-size steps per
# call of World.update(). This should not change things like fish max turn
# radius or fish speed or any perceptible behavior other than to smooth out
# artifacts caused by the discreteness of the simulation.
UPDATE_GRANULARITY : int = 1


## TURBINE POSITIONS/SETTINGS
TURBINE_RADIUS = 10
TURBINE_POSITION = [WORLD_SIZE[0] - 25, WORLD_SIZE[1] / 2, 0]
BLADE_STRIKE_PROBABILITY = 0.11


## ENTRAINMENT/ZOI POSITIONS
ENTRAINMENT_DIMENSIONS = [20, 20, 20]
ZONE_OF_INFLUENCE_DIMENSIONS = [140, 20, 25]
ENTRAINMENT_POSITION = np.array([TURBINE_POSITION[0] + TURBINE_RADIUS - 20, TURBINE_POSITION[1] - 5, 0])
ZONE_OF_INFLUENCE_POSITION = np.array([TURBINE_POSITION[0] + TURBINE_RADIUS - 160, TURBINE_POSITION[1] - 5, 0])


# FISH_BEHAVIOR
COLLISION_AVOIDANCE_DISTANCE = 2
ATTRACTION_DISTANCE = 15
ORIENTATION_DISTANCE = 10
# TRADEOFF BETWEEN ATTRACTION & ORIENTATION
ATTRACTION_WEIGHT = 0.5
MAX_TURN = 0.5  # radians
TURN_NOISE_SCALE = 0.01  # standard deviation in noise
FISH_SPEED = 0.2
FLOW_SPEED = 0.1
FLOW_DIRECTION = np.array([1.0, 0.0, 0.0])
INFORMED_DIRECTION = np.array([1.0, 0.0, 0.0])
INFORMED_DIRECTION_WEIGHT = 0.5
SCHOOLING_WEIGHT = 0.5
# Turbine repulsion behavior. This is technically fish behavior.
TURBINE_REPULSION_STRENGTH = 0.0
TURBINE_EXPONENTIAL_DECAY = -0.1


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

    def has_inside(self, fish):
        return (
                self.position[0] <= fish.position[0] <= self.position[0] + self.dimensions[0] and
                self.position[1] <= fish.position[1] <= self.position[1] + self.dimensions[1])


class World:
    """contains references to all the important stuff in the simulation"""

    def __init__(self):
        self.frame_number = 0
        self.fish_in_ent_count = 0
        self.fish_in_zoi_count = 0
        self.fish_collided_count = 0
        self.fish_struck_count = 0
        self.fish_collided_and_struck_count = 0

        # Initialize fishes.
        self.fishes = []
        for f in range(NUM_FISHES):
            position = np.zeros(DIMENSIONS)
            position[0] = np.random.uniform(10, 200)
            position[1] = np.random.uniform(0, WORLD_SIZE[1])
            position[2] = np.random.uniform(0, WORLD_SIZE[2])
            self.fishes.append(Fish(
                # Position - random, within the WORLD_SIZE of the world
                position,
                # np.random.rand(DIMENSIONS) * WORLD_SIZE,
                # Heading - random, uniform between -1 and 1
                np.random.rand(DIMENSIONS)*2 - 1, world=self, fish_id=f))


        # Initialize both turbines
        # TODO - explain where the turbines are
        self.turbine_base = Turbine(np.array(TURBINE_POSITION), TURBINE_RADIUS, "Base", "red")
        blade_position = np.copy(TURBINE_POSITION)
        blade_position[2] = TURBINE_RADIUS + TURBINE_POSITION[2] # TURBINE_RADIUS * 2
        self.turbine_blade = Turbine(blade_position, TURBINE_RADIUS, "Blade", "red")

        # Initialize both "rectangle"s
        self.entrainment = Rectangle(ENTRAINMENT_POSITION, ENTRAINMENT_DIMENSIONS, "blue")
        self.zone_of_influence = Rectangle(ZONE_OF_INFLUENCE_POSITION, ZONE_OF_INFLUENCE_DIMENSIONS, "blue")


    def update(self):
        """
        Main function of the simulation. Call this to progress the world one
        step forward in time.
        """

        for i in range(UPDATE_GRANULARITY):
            for f in self.fishes:
                f.update()

        # to keep track of the number of frames per simulation
        self.frame_number += 1

        # to keep track of how many fish encounter/interact with each component
        self.fish_in_zoi_count = len([f for f in self.fishes if f.in_zoi])
        self.fish_in_ent_count = len([f for f in self.fishes if f.in_entrainment])
        self.fish_collided_count = len([f for f in self.fishes if f.collided_with_turbine])
        self.fish_struck_count = len([f for f in self.fishes if f.struck_by_turbine])
        self.fish_collided_and_struck_count = len([f for f in self.fishes if f.collided_and_struck])

    def run_full_simulation(self):
        while True:
            self.update()
            if all(fish.left_environment for fish in self.fishes):
                break

    def print_close_out_message(self):
        print("Number of fish in ZOI:", self.fish_in_zoi_count)
        print("Number of fish in entrainment:", self.fish_in_ent_count)
        print("Number of fish collided with the turbine:", self.fish_collided_count)
        print("Number of fish struck by the turbine:", self.fish_struck_count)
        print("Number of fish collided then struck by the turbine:", self.fish_collided_and_struck_count)
        print("Total number of frames in the simulation:", self.frame_number)

        for fish in self.fishes:
            print(f"Fish {fish.id}:")
            print(f"    Frames in Zone of Influence: {fish.fish_in_zoi_frames}")
            print(f"    Frames in Entrainment: {fish.fish_in_ent_frames}")


def distance_between(A, B) -> float:
    return np.linalg.norm(A.position - B.position)


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    else:
        return vector


def turbine_repulsion_strength(distance):
    """Avoidance strength decreases exponentially with distance"""
    avoidance = TURBINE_REPULSION_STRENGTH * math.exp(TURBINE_EXPONENTIAL_DECAY * distance)
    return avoidance


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


class Fish:
    """main agent of the model"""


    # Weighting term is strength between swimming
    # towards desired direction and schooling (1 is all desired direction, 0 is all
    # schooling and ignoring desired direction)

    def __init__(self, position, heading, fish_id, world):
        """initial values for position and heading"""
        self.position = position
        self.heading = heading
        self.color = 'blue'
        self.id = fish_id
        self.all_fish_left = False
        self.left_environment = False
        self.world = world
        self.fish_in_zoi_frames = 0
        self.fish_in_ent_frames = 0

        # Collision/zoi detection variables.
        self.in_zoi = False
        self.in_entrainment = False
        self.collided_with_turbine = False
        self.struck_by_turbine = False
        self.collided_and_struck = False


    def update(self):
        self.update_heading()
        self.move()
        self.check_collisions()


    def desired_heading(self):
        """
        # TODO - let's confirm the priority of these things.

        Rules of desired headings.
        1. Avoid collisions with other fish & turbines (collision_avoidance)
        2. Attract & orient & repel from turbines
        """

        # First find all pairwise distances between us and other fish.
        # TODO - if this is expensive, we can do it once in world and then share it between all fish.
        fish_distances = [(other, distance_between(self, other)) for other in self.world.fishes if other is not self]

        turbine_distances = [
            (self.world.turbine_base, distance_between(self, self.world.turbine_base)),
            (self.world.turbine_blade, distance_between(self, self.world.turbine_blade)),
        ]

        # 1. Avoid collision with other fish & turbines identically.
        # This takes precedence over all other behaviors and should only occur at a very small
        # distance.
        collision_avoidance_found, collision_avoidance_direction = False, np.zeros(DIMENSIONS)
        for other, distance in fish_distances + turbine_distances:
            if distance <= COLLISION_AVOIDANCE_DISTANCE:
                collision_avoidance_found = True
                collision_avoidance_direction += normalize(self.position - other.position)

        if collision_avoidance_found:
            return normalize(collision_avoidance_direction) # EXIT HERE! If we found a collision avoidance, we're done.

        # 2. Attract/align & repel from turbine.

        # Weighted sum of vectors towards other fish within attraction radius and
        # other fishes' headings within orientation distance.
        schooling_direction = np.zeros(DIMENSIONS)
        for other, distance in fish_distances:
            if distance <= ATTRACTION_DISTANCE:
                schooling_direction += ATTRACTION_WEIGHT * normalize(other.position - self.position)
            if distance <= ORIENTATION_DISTANCE:
                schooling_direction += (1.0 - ATTRACTION_WEIGHT) * normalize(other.heading)

        turbine_repulsion_direction = np.zeros(DIMENSIONS)
        for turbine, distance in turbine_distances:
            turbine_repulsion_direction += normalize(self.position - turbine.position) * turbine_repulsion_strength(distance)



        schooling_direction = normalize(schooling_direction)

        # informed direction makes all fish go a specific direction,
        # with an added weight between preferred direction and social behaviors
        # 0 is all social, and 1 is all preferred direction
        # TODO - let's confirm this logic
        desired_heading = (
            INFORMED_DIRECTION_WEIGHT * INFORMED_DIRECTION +
            SCHOOLING_WEIGHT * schooling_direction +
            # The "weight" of this is controlled more directly by the exponential strength
            # function. We *don't* want to normalize it and then reweigh it, or we lose the
            # exponential decay aspect and just have the desired direction.
            turbine_repulsion_direction
        )

        return normalize(desired_heading)


    def update_heading(self):
        desired_heading = self.desired_heading()

        """Assumes self.heading and desired_heading are unit vectors"""
        if desired_heading is not None:
            # Generating some random noise to the heading
            # TODO - this might lead to less noise when there's higher update granularity.
            noise = np.random.normal(0, TURN_NOISE_SCALE/UPDATE_GRANULARITY, len(desired_heading))
            noisy_new_heading = desired_heading + noise

            dot = np.dot(noisy_new_heading, self.heading)
            dot = min(1.0, dot)
            dot = max(-1.0, dot)
            angle_between = np.arccos(dot)
            max_turn_per_update = MAX_TURN / UPDATE_GRANULARITY

            if angle_between > max_turn_per_update:
                noisy_new_heading = rotate_towards(self.heading, noisy_new_heading, max_turn_per_update)

            self.heading = noisy_new_heading


    def move(self):
        self.position += self.heading * FISH_SPEED / UPDATE_GRANULARITY

        # Applies circular boundary conditions to y.
        self.position[1] = self.position[1] % WORLD_SIZE[1]

        # Apply "reflective" boundary conditions to the z.
        if not 0 <= self.position[2] <= WORLD_SIZE[2]:
            self.heading[2] = -self.heading[2]
            self.position[2] = np.clip(self.position[2], 0, WORLD_SIZE[2])

        # (In the x direction - they can go off the edge of the world)

        # adding flow to fish's position including the FISH_speed and direction
        # fish are unaware of flow
        # Flow vector is always 1.0 in the x direction; zero in other directions
        self.position += FLOW_SPEED * FLOW_DIRECTION / UPDATE_GRANULARITY



    def check_collisions(self):

        if self.position[0] < 0 or self.position[0] > WORLD_SIZE[0]:
            self.left_environment = True

        if self.world.zone_of_influence.has_inside(self):
            self.in_zoi = True
            self.fish_in_zoi_frames += 1

        if self.world.entrainment.has_inside(self):
            self.in_entrainment = True
            self.fish_in_ent_frames += 1

        if distance_between(self, self.world.turbine_base) <= self.world.turbine_base.radius:
            self.collided_with_turbine = True

        if distance_between(self, self.world.turbine_blade) <= self.world.turbine_blade.radius:
            if np.random.rand() <= BLADE_STRIKE_PROBABILITY:
                self.struck_by_turbine = True

                # if fish previously collided with the turbine
                if self.collided_with_turbine:
                    self.collided_and_struck = True