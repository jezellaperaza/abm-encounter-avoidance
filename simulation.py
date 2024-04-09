from __future__ import annotations
import numpy as np

## WORLD PARAMETERS
NUM_FISHES = 100
WORLD_SIZE = (400, 100, 55)
BURN_IN_FACTOR = 100
BURN_IN_LENGTH = BURN_IN_FACTOR * NUM_FISHES ** (1 / 3)
BURN_IN_WORLD_SIZE = (100, 100, 55)
BURN_IN_TIME = 70  # about 5% of the total runtime
DIMENSIONS = len(WORLD_SIZE)
# If this is greater than 1, (say 5), we'll make 5 mini 1/5-size steps per
# call of World.update(). This should not change things like fish max turn
# radius or fish speed or any perceptible behavior other than to smooth out
# artifacts caused by the discreteness of the simulation.
UPDATE_GRANULARITY: int = 1

## TURBINE POSITIONS/SETTINGS

TURBINE_BASE_RADIUS = 10
TURBINE_BASE_HEIGHT = 15
TURBINE_BASE_CENTER = [WORLD_SIZE[0] - 25, WORLD_SIZE[1] / 2, 0]

TURBINE_BLADE_RADIUS : float = 15
TURBINE_BLADE_HEIGHT : float = 2
TURBINE_BLADE_COLOR = "red"

BLADE_STRIKE_PROBABILITY = 0.11

## ENTRAINMENT/ZOI POSITIONS
ENTRAINMENT_DIMENSIONS = [30, 30, 30]
ZONE_OF_INFLUENCE_DIMENSIONS = [140, 30, 25]
ENTRAINMENT_POSITION = np.array([TURBINE_BASE_CENTER[0] + TURBINE_BASE_RADIUS - 20, TURBINE_BASE_CENTER[1] - 5, 0])
ZONE_OF_INFLUENCE_POSITION = np.array([TURBINE_BASE_CENTER[0] + TURBINE_BASE_RADIUS - 160, TURBINE_BASE_CENTER[1] - 5, 0])

# FISH_BEHAVIOR
COLLISION_AVOIDANCE_DISTANCE = 2.0
TURBINE_AVOIDANCE_DISTANCE = 10
ATTRACTION_DISTANCE = 15
ORIENTATION_DISTANCE = 10
# TRADEOFF BETWEEN ATTRACTION & ORIENTATION
ATTRACTION_WEIGHT = 0.2
MAX_TURN = 0.8  # radians
TURN_NOISE_SCALE = 0.01  # standard deviation in noise
FISH_SPEED = 1.0
FLOW_SPEED = 0.5
FLOW_DIRECTION = np.array([1.0, 0.0, 0.0])
INFORMED_DIRECTION = np.array([1.0, 0.0, 0.0])
INFORMED_DIRECTION_WEIGHT = 0.2
SCHOOLING_WEIGHT = 0.5
# Turbine repulsion behavior. This is technically fish behavior.
TURBINE_REPULSION_STRENGTH = 1.0
TURBINE_EXPONENTIAL_DECAY = 0.2


class TurbineBlade:
    """
    TurbineBlade is a vertical cylinder aligned along the
    y, z plane – i.e. vectors in the x direction are perpendicular to the
    face/base of the cylinder.

    The only parameters controlled externally should be the radius and the
    "height" - the center position should be such that the blade is perfectly
    stacked on top of the cylinder.
    """
    def __init__(self):
        self.center = np.array(TURBINE_BASE_CENTER) += np.array([0, 0, 1]) * (TURBINE_BASE_HEIGHT / 2.0 + TURBINE_BLADE_RADIUS)
        self.position = self.center
        self.radius = TURBINE_BLADE_RADIUS
        self.height = TURBINE_BLADE_HEIGHT
        self.color = TURBINE_BLADE_COLOR

    def has_inside(self, fish):
        return self.distance_to_fish_raw(fish) <= 0.0

    def distance_to_fish_raw(self, fish):
        """
        Returns distance between a fish and the closest surface of the cylinder.
        It can return a negative number (if the fish is inside the cylinder).
        """
        f_x, f_y, f_z = fish.position
        c_x, c_y, c_z = self.center

        # Radial distance in the y, z plane
        radial_distance = np.sqrt((f_y - c_y)**2 + (f_z - c_z)**2)

        # Distance in the x direction
        face_distance = np.abs(f_x - c_x) - self.height / 2


        # If we're in the infinite cylinder, then it's just distance to the face of it
        if radial_distance <= self.radius:
            return face_distance
        # Otherwise we do the pythagorean distance. This either gives the 
        # distance to the nearest edge of the cylinder, or just gives the radial
        # distance if we're right over it.
        else:
            return np.sqrt(math.floor(face_distance, 0)**2 + radial_distance**2)

    def distance_to_fish(self, fish):
        return max(self.distance_to_fish_raw(fish), 0)




def inside_cylinder(base_center, radius, height, point):
    """
    inside_cylinder returns whether point is inside the cylinder defined
    by base_center, radius and height
    """
    # check if fish's x, y coordinates are inside the infinitely tall cylinder
    inside_cylinder = np.linalg.norm(point[0:2] - base_center[0:2]) <= radius

    # check if fish's z coordinates are above the base and below the top
    z_inside = base_center[2] <= point[2] <= base_center[2] + height
    return inside_cylinder and z_inside


class TurbineBase:
    def __init__(self, base_center, height, radius):
        self.base_center = base_center
        self.height = height
        self.radius = radius
        # position of turbine for avoidance is defined as the center - halfway up from the base
        self.position = self.base_center + np.array([0, 0, height / 2.0])

    def has_inside(self, fish):
        return inside_cylinder(self.base_center, self.radius, self.height, fish.position)


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
        self.burn_in = True

        # Initialize fishes.
        self.fishes = []
        self.burn_in_positions = []

        min_bound = [0, 0, 0]
        max_bound = [BURN_IN_WORLD_SIZE[i] - BURN_IN_LENGTH if BURN_IN_LENGTH < BURN_IN_WORLD_SIZE[i] else 0 for i in
                     range(DIMENSIONS)]
        burn_in_placement = [np.random.randint(min_bound[i], max_bound[i] + 1) for i in range(DIMENSIONS)]

        for f in range(NUM_FISHES):
            initial_position = np.zeros(DIMENSIONS)

            # Initial positions of fish within the cube
            initial_position[0] = np.random.uniform(0, BURN_IN_LENGTH)
            initial_position[1] = np.random.uniform(0, BURN_IN_LENGTH)
            initial_position[2] = np.random.uniform(0, BURN_IN_LENGTH)

            # Initial positions of fish within the cube
            # initial_position[0] = np.random.uniform(0, BURN_IN_WORLD_SIZE[0])
            # initial_position[1] = np.random.uniform(0, BURN_IN_WORLD_SIZE[1])
            # initial_position[2] = np.random.uniform(0, BURN_IN_WORLD_SIZE[2])

            # Save initial positions
            self.burn_in_positions.append(initial_position + burn_in_placement)

            self.fishes.append(Fish(
                initial_position + burn_in_placement,
                np.random.rand(DIMENSIONS) * 2 - 1,
                world=self,
                fish_id=f))

        # Initialize both turbines
        self.turbine_base = TurbineBase(np.array(TURBINE_BASE_CENTER), TURBINE_BASE_HEIGHT, TURBINE_BASE_RADIUS)
        self.turbine_blade = TurbineBlade()

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
        # print(f'\r{self.frame_number}', end='')

        if self.burn_in and self.frame_number > BURN_IN_TIME:
            self.burn_in = False
            # print("\nBurn in complete.")

        # to keep track of how many fish encounter/interact with each component
        self.fish_in_zoi_count = len([f for f in self.fishes if f.in_zoi])
        self.fish_in_ent_count = len([f for f in self.fishes if f.in_entrainment])
        self.fish_collided_count = len([f for f in self.fishes if f.collided_with_turbine_base])
        self.fish_struck_count = len([f for f in self.fishes if f.struck_by_turbine_blade])
        self.fish_collided_and_struck_count = len([f for f in self.fishes if f.collided_with_turbine_blade and f.struck_by_turbine_blade])

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
        # for fish in self.fishes:
        #     print(f"Fish {fish.id}:")
        #     print(f"    Frames in Zone of Influence: {fish.fish_in_zoi_frames}")
        #     print(f"    Frames in Entrainment: {fish.fish_in_ent_frames}")


def adjust_B_for_y_periodicity(A, B):
    other_position = np.copy(B.position)
    world_y_length = WORLD_SIZE[2]
    if A.world.burn_in:
        world_y_length = BURN_IN_WORLD_SIZE[2]

    if np.linalg.norm(other_position[2] - A.position[2]) > world_y_length:
        if other_position[2] > A.position[2]:
            other_position[2] = other_position[2] - world_y_length
        else:
            other_position[2] = other_position[2] + world_y_length
    return other_position


def distance_between(A, B) -> float:
    other_position = adjust_B_for_y_periodicity(A, B)
    return np.linalg.norm(A.position - other_position)


def direction_towards(A, B) -> float:
    # This has to take into account periodic conditions in the y direction only
    other_position = adjust_B_for_y_periodicity(A, B)

    return normalize(other_position - A.position)


def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    else:
        return vector


def turbine_repulsion_strength(distance):
    """Avoidance strength decreases exponentially with distance"""
    # if distance >= TURBINE_AVOIDANCE_DISTANCE:
    #     return 0.0
    assert TURBINE_EXPONENTIAL_DECAY > 0
    avoidance = TURBINE_REPULSION_STRENGTH * np.exp(-1 * TURBINE_EXPONENTIAL_DECAY * distance)
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
        self.collided_with_turbine_base = False
        self.struck_by_turbine_blade = False
        self.collided_and_struck = False

    def update(self):
        self.update_heading()
        self.move()
        self.check_collisions()

    def desired_heading(self):
        """
        Rules of desired headings.
        1. Avoid collisions with other fish & turbines (collision_avoidance)
        2. Attract & orient & repel from turbines
        """

        # First find all pairwise distances between us and other fish.
        # TODO - if this is expensive, we can do it once in world and then share it between all fish.
        fish_distances = [(other, distance_between(self, other)) for other in self.world.fishes if other is not self]

        turbine_distances = [
            (self.world.turbine_base, distance_between(self, self.world.turbine_base)),
            (self.world.turbine_blade, self.world.turbine_blade.distance_to_fish(self)),
        ]

        # 0. Avoid collision with other fish.
        # This takes precedence over all other behaviors and should only occur at a very small
        # distance.
        collision_avoidance_found, collision_avoidance_direction = False, np.zeros(DIMENSIONS)
        for other, distance in fish_distances:
            if distance <= COLLISION_AVOIDANCE_DISTANCE:
                collision_avoidance_found = True
                collision_avoidance_direction += -1 * direction_towards(self, other)

        if collision_avoidance_found:
            return normalize(collision_avoidance_direction)  # EXIT HERE! If we found a collision avoidance, we're done.

        # 1. Avoid collision with turbines.
        turbine_avoidance_found, turbine_avoidance_direction = False, np.zeros(DIMENSIONS)
        for other, distance in turbine_distances:
            if distance <= TURBINE_AVOIDANCE_DISTANCE:
                relative_position = self.position - other.position
                turbine_avoidance_found = True
                turbine_avoidance_direction += normalize(relative_position) / distance  # making it relative to distance

        if turbine_avoidance_found:
            return normalize(turbine_avoidance_direction)  # EXIT HERE! If we found a collision avoidance, we're done.

        # 2. Attract/align & repel from turbine.

        # Weighted sum of vectors towards other fish within attraction radius and
        # other fishes' headings within orientation distance.
        schooling_direction = np.zeros(DIMENSIONS)
        for other, distance in fish_distances:
            if distance <= ATTRACTION_DISTANCE:
                schooling_direction += ATTRACTION_WEIGHT * direction_towards(self, other)
            if distance <= ORIENTATION_DISTANCE:
                schooling_direction += (1.0 - ATTRACTION_WEIGHT) * normalize(other.heading)

        turbine_repulsion_direction = np.zeros(DIMENSIONS)
        for turbine, distance in turbine_distances:
            relative_position = self.position - turbine.position
            turbine_repulsion_direction += normalize(relative_position) * turbine_repulsion_strength(distance)
            # print(turbine_repulsion_direction)

        schooling_direction = normalize(schooling_direction)

        # informed direction makes all fish go a specific direction,
        # with an added weight between preferred direction and social behaviors
        # 0 is all social, and 1 is all preferred direction

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
            noise = np.random.normal(0, TURN_NOISE_SCALE / UPDATE_GRANULARITY, len(desired_heading))
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

        # for i in range(DIMENSIONS):
        #     self.position[i] %= WORLD_SIZE[i]

        if self.world.burn_in:
            # Apply periodic boundaries for the burn-in region
            for i in range(DIMENSIONS):
                self.position[i] %= BURN_IN_WORLD_SIZE[i]

        else:
            # Applies circular boundary conditions to y.
            self.position[1] = self.position[1] % WORLD_SIZE[1]
            self.position[2] = np.clip(self.position[2], 0, WORLD_SIZE[2])

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

        if self.world.turbine_base.has_inside(self):
            self.collided_with_turbine_base = True

        if self.world.turbine_blade.has_inside(self):
            if np.random.rand() <= BLADE_STRIKE_PROBABILITY:
                self.struck_by_turbine_blade = True
