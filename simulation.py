from __future__ import annotations
import numpy as np
import math


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


class World():
    """contains references to all the important stuff in the simulation"""

    NUM_FISHES = 50
    SIZE = (100, 100, 100)
    # Specifies the number of dimensions in the simulation
    # If 2, then the dimensions are [X, Y]
    # If 3, then the dimensions are [X, Y, Z]
    DIMENSIONS = 3
    TURBINE_RADIUS = 5
    TURBINE_POSITION = [SIZE[0] - 25, SIZE[1] / 2, 0]
    ENTRAINMENT_DIMENSIONS = [10, 10, 10]
    ZONE_OF_INFLUENCE_DIMENSIONS = [140, 10, 25]
    ENTRAINMENT_POSITION = np.array([TURBINE_POSITION[0] + TURBINE_RADIUS - 20, TURBINE_POSITION[1] - 5, 0])
    ZONE_OF_INFLUENCE_POSITION = np.array([TURBINE_POSITION[0] + TURBINE_RADIUS - 160, TURBINE_POSITION[1] - 5, 0])
    # TIME_FRAME = 100
    UPDATES_PER_TIME = 10


    def __init__(self):
        self.frame_number = 0


        # Initialize fishes.
        self.fishes = []
        for f in range(World.NUM_FISHES):
            self.fishes.append(Fish(
                # Position - random, within the SIZE of the world
                np.random.rand(World.DIMENSIONS) * World.SIZE,
                # Heading - random, uniform between -1 and 1
                np.random.rand(World.DIMENSIONS)*2 - 1, world=self, fish_id=f))

        # Initialize both turbines
        self.turbine_base = Turbine(np.array(World.TURBINE_POSITION), World.TURBINE_RADIUS, "Base", "red")
        blade_position = World.TURBINE_POSITION
        # TODO - Not sure why this is happening.
        # TODO - Jezella: if I'm understanding this new change correctly, this is splitting the turbine in half to be top and bottom so bottom is collision and top is strike
        # TODO - is there a more intuitive wait to fix this?
        blade_position[2] = World.TURBINE_RADIUS * 2
        self.turbine_blade = Turbine(blade_position, World.TURBINE_RADIUS, "Blade", "red")

        # Initialize both "rectangle"s
        self.entrainment = Rectangle(World.ENTRAINMENT_POSITION, World.ENTRAINMENT_DIMENSIONS, "blue")
        self.zone_of_influence = Rectangle(World.ZONE_OF_INFLUENCE_POSITION, World.ZONE_OF_INFLUENCE_DIMENSIONS, "blue")



    def update(self):
        """
        Main function of the simulation. Call this to progress the world one
        step forward in time.
        """
        for f in self.fishes:
            f.update()

        self.frame_number += 1


def distance_between(A, B) -> float:
    return np.linalg.norm(A.position - B.position)


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    else:
        return vector


def avoidance_strength(distance):
    """Avoidance strength decreases exponentially with distance"""
    k = -0.05
    repulsion_strength_at_zero = 1
    avoidance = repulsion_strength_at_zero * math.exp(k * distance)
    return avoidance
    # TODO - why was avoidance at most 0? This seems like a bug.
    # TODO - Jezella: I might have been thinking didn't want any additional things to happen - so just cap it at zero (meaning no avoidance).
    #return max(0.0, avoidance)


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
    ATTRACTION_DISTANCE = 25
    ORIENTATION_DISTANCE = 15
    ATTRACTION_WEIGHT = 0.5
    MAX_TURN = 0.1  # radians
    TURN_NOISE_SCALE = 0.1  # standard deviation in noise
    # TODO - Jezella: what if fish are 0.155 m in length, and I want them swimming 1-body length per second?
    FISH_SPEED = 0.2
    FLOW_SPEED = 0
    REACTION_DISTANCE = 10
    BLADE_STRIKE_PROBABILITY = 0.11

    # Weighting term is strength between swimming
    # towards desired direction and schooling (1 is all desired direction, 0 is all
    # schooling and ignoring desired direction)
    INFORMED_DIRECTION_WEIGHT = 0 
    INFORMED_DIRECTION = np.array([1.0, 0.0, 0.0])

    def __init__(self, position, heading, fish_id, world):
        """initial values for position and heading"""
        self.position = position
        self.heading = heading
        self.color = 'blue'
        self.id = fish_id
        self.all_fish_left = False
        self.left_environment = False
        self.world = world

        # Collision/zoi dection variables.
        self.in_zoi = False
        self.in_entrainment = False
        self.collided_with_turbine = False
        self.struck_by_turbine = False


    def update(self):
        self.update_heading()
        self.move()
        self.check_collisions()


    def desired_heading(self):
        """
        # TODO - let's confirm the priority of these things.

        Rules of desired headings.
        1. Avoid collisions with other fish (repulsion)
        2. Avoid collisions with turbines (avoidance)
        2. Attract & orient
        """
        desired_heading = np.zeros(self.world.DIMENSIONS)

        # First find all pairwise distances between us and other fish.
        # TODO - if this is expensive, we can do it once in world and then share it between all fish.
        fish_distances = [(other, distance_between(self, other)) for other in self.world.fishes if other is not self]

        # 1. Repulse from other fish
        repulsion_found = False
        for other, distance in fish_distances:
            if distance <= Fish.REPULSION_DISTANCE:
                repulsion_found = True
                desired_heading += normalize(self.position - other.position)
        if repulsion_found:
            return normalize(desired_heading) # EXIT HERE! If we found a repulsion, we're done.


        # 2. Avoid turbines
        # TODO - there were some wonky bugs in this logic previously I'm pretty sure.
        turbine_distances = [
            (self.world.turbine_base, distance_between(self, self.world.turbine_base)),
            (self.world.turbine_blade, distance_between(self, self.world.turbine_blade)),
        ]
        avoidance_found = False
        for turbine, distance in turbine_distances:
            if distance <= Fish.REACTION_DISTANCE:
                avoidance_found = True
                desired_heading += normalize(self.position - turbine.position)

        if avoidance_found:
            return normalize(desired_heading) # EXIT HERE! If we found an avoidance, we're done.

        # 3. Attract/align.
        #
        # Weighted sum of vectors towards other fish within attraction radius and
        # other fishes' headings within orientation distance.
        for other, distance in fish_distances:
            if distance <= Fish.ATTRACTION_DISTANCE:
                desired_heading += Fish.ATTRACTION_WEIGHT * normalize(other.position - self.position)
            if distance <= Fish.ORIENTATION_DISTANCE:
                desired_heading += (1.0 - Fish.ATTRACTION_WEIGHT) * normalize(other.heading)

        desired_heading = normalize(desired_heading)

        # informed direction makes all fish go a specific direction,
        # with an added weight between preferred direction and social behaviors
        # 0 is all social, and 1 is all preferred direction
        # TODO - let's confirm this logic
        # TODO - Jezella: Yeah, with this I'm assuming we won't have that informed direction/schooling trade-off? Which might be fine, ask Andrew.
        desired_heading = Fish.INFORMED_DIRECTION_WEIGHT * Fish.INFORMED_DIRECTION + (1.0 - Fish.INFORMED_DIRECTION_WEIGHT) * self.heading

        return normalize(desired_heading)



    def update_heading(self):
        new_heading = self.desired_heading()

        """Assumes self.heading and new_heading are unit vectors"""
        if new_heading is not None:
            # Generating some random noise to the fish.heading
            noise = np.random.normal(0, Fish.TURN_NOISE_SCALE, len(new_heading))
            noisy_new_heading = new_heading + noise

            dot = np.dot(noisy_new_heading, self.heading)
            dot = min(1.0, dot)
            dot = max(-1.0, dot)
            angle_between = np.arccos(dot)
            max_turn_per_update = Fish.MAX_TURN / World.UPDATES_PER_TIME

            if angle_between > max_turn_per_update:
                noisy_new_heading = rotate_towards(self.heading, noisy_new_heading, max_turn_per_update)

            self.heading = noisy_new_heading


    def move(self):
        # self.position += (self.heading * Fish.FISH_SPEED)
        # TODO - Jezella: Want to double check this updates per time option. Is it more important for GIFs? Can see difference
        # TODO - between update_per_time = 1 and 10.
        velocity = self.heading * Fish.FISH_SPEED
        new_position = velocity / World.UPDATES_PER_TIME
        self.position += new_position

        # Applies circular boundary conditions to y and z but not x.
        # TODO - we were previously applying this to all dimensions so we could never leave the world.
        # TODO - Jezella: Want to confirm options of all periodic boundaries or some. Talk this through.
        self.position[1:] = self.position[1:] % World.SIZE[1:]

        # adding flow to fish's position including the speed and direction
        # fish are unaware of flow
        # Flow vector is always 1.0 in the x direction; zero in other directions
        flow_vector = np.zeros(World.DIMENSIONS)
        flow_vector[0] = 1.0
        self.position += self.FLOW_SPEED * flow_vector



    def check_collisions(self):


        if self.position[0] < 0 or self.position[0] > World.SIZE[0]:
            self.left_environment = True

        if self.world.zone_of_influence.has_inside(self):
            self.in_zoi = True

        if self.world.entrainment.has_inside(self):
            self.in_entrainment = True

        if distance_between(self, self.world.turbine_base) <= self.world.turbine_base.radius:
            self.collided_with_turbine = True

        if distance_between(self, self.world.turbine_blade) <= self.world.turbine_blade.radius:
            # TODO - this logic didn't make sense to me before.
            # It was basically saying if a random int was between 0.002 and 0.013, then we got struck.
            # Why not just say, if a random number is <= 0.011?
            # TODO - Jezella: I was trying to base this off some of the published literature we have. But if causes problems - can work with.
            if np.random.rand() <= fish.BLADE_STRIKE_PROBABILITY:
                self.struck_by_turbine = True


