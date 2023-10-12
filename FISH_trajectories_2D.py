from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.patches as patches

class World():
	"""contains references to all the important stuff in the simulation"""

	NUM_FISHES = 15
	SIZE = 100

	def __init__(self):
		self.fishes: list[Fish] = []
		self.turbines: list[Turbine] = []

	def add_turbine(self, position, color='red'):
		turbine = Turbine(position, color)
		self.turbines.append(turbine)

# class Turbine():
# 	def __init__(self, position, color='red'):
# 		self.position = np.array(position)
# 		self.color = color

class Turbine:
    def __init__(self, points, color='red'):
        self.points = points
        self.color = color

def distance_between(fishA: Fish, fishB: Fish) -> float:
	return np.linalg.norm(fishA.position - fishB.position)

# for determining fish that are within model component boxes
def fish_within(point, vertices):
	x, y = point
	n = len(vertices)
	inside = False

	for i in range(n):
		j = (i + 1) % n
		xi, yi = vertices[i]
		xj, yj = vertices[j]

		if yi <= y < yj or yj <= y < yi:
			if x > (xj - xi) * (y - yi) / (yj - yi) + xi:
				inside = not inside

	return inside

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
	turbine_repulsion_found = False
	repulsion_direction = np.array([0.0, 0.0])
	avoidance_direction = np.array([0.0, 0.0])
	avoidance_probability = 0.4

	for other, distance in others:
		if distance <= Fish.REPULSION_DISTANCE:
			repulsion_found = True
			vector_difference = other.position - fish.position
			repulsion_direction -= (vector_difference / np.linalg.norm(vector_difference))

	if repulsion_found:
		return repulsion_direction / np.linalg.norm(repulsion_direction)

	# using the same strategy as the repulsion between fish and its neighbors, I am
	# trying to repulse fish and the turbine
	# the strength between the two should depend on the inverse proportion to the distance,
	# the closer fish are to the turbine, the more immediate and abrupt the avoidance is
	# the reaction distance is set to 15 as an arbitrary number

	for turbine in world.turbines:

		if turbine.color == 'red':
			for i in range(4):
				vector_to_fish = fish.position - turbine.points[i]
				distance_to_turbine = np.linalg.norm(vector_to_fish)
				if distance_to_turbine <= 15 and np.random.rand() > avoidance_probability:
					avoidance_strength = 1 / distance_to_turbine
					avoidance_direction += (vector_to_fish / distance_to_turbine) * avoidance_strength
					turbine_repulsion_found = True

		# This section is to mimic the collisions occurring between fish and the turbine
		# right now the code is set for fish to bounce back by applying their heading to -1
		# could potentially keep the same or do a reflection equation
		if turbine.color == 'red':
			turbine_left_x = min(p[0] for p in turbine.points)
			turbine_right_x = max(p[0] for p in turbine.points)
			turbine_bottom_y = min(p[1] for p in turbine.points)
			turbine_top_y = max(p[1] for p in turbine.points)

			if (fish.position[0] >= turbine_left_x and fish.position[0] <= turbine_right_x and
					fish.position[1] >= turbine_bottom_y and fish.position[1] <= turbine_top_y):
				fish.heading *= -1

	if turbine_repulsion_found:
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

	# if fish are informed, an informed direction is calculated by multiplying the direction and weight
	# the new informed direction is applied to the attraction/alignment direction where 1 is informed_fish are
	# considering only the desired direction and 0 is informed_fish ignore their desired direction and resume
	# schooling behaviors
	if fish.informed:
		informed_direction = Fish.DESIRED_DIRECTION * Fish.DESIRED_DIRECTION_WEIGHT
		attraction_orientation_direction = informed_direction + (1 - Fish.DESIRED_DIRECTION_WEIGHT) * attraction_orientation_direction

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
	NUM_FRAMES = 30
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
	FLOW_SPEED = 3

	def __init__(self, position, heading, informed=False):
		"""initial values for position and heading
		setting up the informed fish from a subset of NUM_FISHES
		pink fish are informed, but fish are uninformed"""
		self.position = position
		self.heading = heading
		self.informed = informed
		self.color = plt.cm.viridis(np.random.rand())

	def move(self):
		self.position += self.heading * Fish.SPEED

		# Applies circular boundary conditions without worrying about
		# heading decisions.
		self.position = np.mod(self.position, World.SIZE)

	def update_heading(self, new_heading):
		"""Assumes self.heading and new_heading are unit vectors"""

		if new_heading is not None:

			# adding flow to fish's movement including the speed and direction
			# need to figure out how to update fish's collision immediately
			# when flow is in place
			new_heading += Fish.FLOW_VECTOR * Fish.FLOW_SPEED

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
    fish_in_zoi = set()
    fish_in_ent = set()

    world.add_turbine([(60, 50), (70, 50), (70, 60), (60, 60)], color='red')
    world.add_turbine([(50, 50), (60, 50), (60, 60), (50, 60)], color='blue')
    world.add_turbine([(50, 60), (50, 50), (20, 50), (20, 60)], color='green')

    for f in range(World.NUM_FISHES):
        initial_position = np.array([np.random.uniform(0, 10), np.random.rand() * World.SIZE])
        world.fishes.append(Fish(initial_position, np.random.rand(2), informed=True if f < 10 else False))

    fig, ax = plt.subplots()
    turbine_patches = [
        patches.Polygon(turbine.points, edgecolor=turbine.color, facecolor='none')
        for turbine in world.turbines
    ]

    for patch in turbine_patches:
        ax.add_patch(patch)
    plt.xlim(0, World.SIZE)
    plt.ylim(0, World.SIZE)

    for _ in range(Fish.NUM_FRAMES):

        for f_num, f in enumerate(world.fishes):
            for turbine in world.turbines:
                if turbine.color == 'green' and fish_within(f.position, turbine.points):
                    fish_in_zoi.add(f_num)

                if turbine.color == 'blue' and fish_within(f.position, turbine.points):
                    fish_in_ent.add(f_num)

        x = [f.position[0] for f in world.fishes]
        y = [f.position[1] for f in world.fishes]
        colors = [f.color for f in world.fishes]

        plt.scatter(x, y, s=5, c=colors)

        for f in world.fishes:
            f.update_heading(desired_new_heading(f, world))
        for f in world.fishes:
            f.move()

    plt.show()

main()