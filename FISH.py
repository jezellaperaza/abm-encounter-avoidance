from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.patches as patches

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

class Turbine():
	def __init__(self, position, color='red'):
		self.position = np.array(position)
		self.color = color

def distance_between(fishA: Fish, fishB: Fish) -> float:
	return np.linalg.norm(fishA.position - fishB.position)

def desired_new_heading(fish: Fish, world: World):

	# find all pairwise distances
	others: list[(Fish, float)] = []
	turbines: list[(Turbine, float)] = []

	for other in world.fishes:
		if other is not fish:
			others.append((other, distance_between(fish, other)))

	for turbine in world.turbines:
		turbines.append((turbine, distance_between(fish, turbine)))

	# Compute repulsion

	# use this to make sure we're not messing with float comparison to decide
	# whether we had something inside the repulsion distance:
	repulsion_found = False
	turbine_repulsion_found = False
	repulsion_direction = np.array([0.0, 0.0])
	turbine_repulsion_direction = np.array([0.0, 0.0])

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
	for turbine, distance in turbines:
		if distance <= Fish.REPULSION_DISTANCE_FROM_TURBINE:
			turbine_repulsion_found = True
			turbine_strength = 1.0 / distance  # this is the inverse proportion to distance
			turbine_vector_difference = turbine.position - fish.position
			turbine_repulsion_direction -= (turbine_strength * turbine_vector_difference / np.linalg.norm(turbine_vector_difference))

	# This section is to mimic the collisions occurring between fish and the turbine
	# right now the code is set for fish to bounce back by applying their heading to -1
	# could potentially keep the same or do a reflection equation
	# for turbine, _ in turbines:
	# 	if turbine.color == 'red':
	# 		turbine_left_x = min(p[0] for p in turbine.position)
	# 		turbine_right_x = max(p[0] for p in turbine.position)
	# 		turbine_bottom_y = min(p[1] for p in turbine.position)
	# 		turbine_top_y = max(p[1] for p in turbine.position)
	#
	# 		if (fish.position[0] >= turbine_left_x and fish.position[0] <= turbine_right_x and
	# 			fish.position[1] >= turbine_bottom_y and fish.position[1] <= turbine_top_y):
	# 			fish.heading *= -1

	if turbine_repulsion_found:
		return turbine_repulsion_direction / np.linalg.norm(turbine_repulsion_direction)

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
	REPULSION_DISTANCE = 1
	ATTRACTION_DISTANCE = 15
	ORIENTATION_DISTANCE = 10
	REPULSION_DISTANCE_FROM_TURBINE = 20
	ATTRACTION_ALIGNMENT_WEIGHT = 0.5
	MAX_TURN = 0.1
	TURN_NOISE_SCALE = 0.1 # standard deviation in noise
	SPEED = 1.0
	DESIRED_DIRECTION = np.array([1, 0]) # Desired direction of informed fish is towards the right when [1, 0]
	DESIRED_DIRECTION_WEIGHT = 0.8 # Weighting term is strength between swimming
									# towards desired direction and schooling (1 is all desired direction, 0 is all
									# schooling and ignoring desired ditrection

	def __init__(self, position, heading, informed=False):
		"""initial values for position and heading
		setting up the informed fish from a subset of NUM_FISHES
		pink fish are informed, but fish are uninformed"""
		self.position = position
		self.heading = heading
		self.informed = informed
		self.color = 'pink' if informed else 'blue'

	def move(self):
		self.position += self.heading * Fish.SPEED

		# Applies circular boundary conditions without worrying about
		# heading decisions.
		self.position = np.mod(self.position, World.SIZE)

	def update_heading(self, new_heading):
		"""Assumes self.heading and new_heading are unit vectors"""

		if new_heading is not None:

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

	world.add_turbine([(60, 40), (80, 40), (80, 60), (60, 60)], color='red') # turbine placement bottom-left, bottom-right, top-right, top-left

	for f in range(10):
		world.fishes.append(Fish((np.random.rand(2)) * World.SIZE, np.random.rand(2), informed=True))
		# initial_position = np.array([np.random.uniform(0, 10), np.random.rand() * World.SIZE])
		# world.fishes.append(Fish(initial_position, np.random.rand(2), informed=True)) # Subsets the informed fish
																					# and makes them start to the left of environment

	for f in range(World.NUM_FISHES - 10):
		world.fishes.append(Fish((np.random.rand(2)) * World.SIZE, np.random.rand(2), informed=False))
		# initial_position = np.array([np.random.uniform(0, 10), np.random.rand() * World.SIZE])
		# world.fishes.append(Fish(initial_position, np.random.rand(2), informed=False)) # The remaining fish that are not informed are
																					# also set to the left of the environment
	# for f in range(World.NUM_FISHES):
	# 	world.fishes.append(Fish((np.random.rand(2))*World.SIZE, np.random.rand(2)))

	fig, ax = plt.subplots()
	x, y = [],[]
	sc = ax.scatter(x,y)

	turbine_patches = [
		patches.Polygon(turbine.position, edgecolor=turbine.color, facecolor='none')
		for turbine in world.turbines
	]

	for patch in turbine_patches:
		ax.add_patch(patch)
	plt.xlim(0, World.SIZE)
	plt.ylim(0, World.SIZE)

	def animate(_):
		x = [f.position[0] for f in world.fishes]
		y = [f.position[1] for f in world.fishes]
		sc.set_offsets(np.c_[x,y])
		for f in world.fishes:
			f.update_heading(desired_new_heading(f, world))
		for f in world.fishes:
			f.move()

		colors = [f.color for f in world.fishes]
		sc.set_color(colors)

	ani = matplotlib.animation.FuncAnimation(fig, animate,
	                frames=2, interval=100, repeat=True)
	plt.show()

main()