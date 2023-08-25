from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation

class World():
	"""contains references to all the important stuff in the simulation"""

	NUM_FISHES = 64
	SIZE = 100

	def __init__(self):
		self.fishes: list[Fish] = []

def distance_between(fishA: Fish, fishB: Fish) -> float:
	return np.linalg.norm(fishA.position - fishB.position)

def desired_new_heading(fish: Fish, world: World):

	if fish.informed:
		fish.DESRIRED_DIRECTION_WEIGHT * fish.DESIRED_DIRECTION

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

	# If we didn't find anything within the repulsion distance, then we
	# do attraction distance and orientation distance.
	# It's an unweighted sum of all the unit vectors:
	# + pointing towards other fish inside ATTRACTION_DISTANCE
	# + pointing in the same direction as other fish inside ORIENTATION_DISTANCE
	attraction_orientation_found = False
	attraction_orientation_direction = np.array([0.0, 0.0])
	for other, distance in others:
		if distance <= Fish.ATTRACTION_DISTANCE:
			attraction_orientation_found = True
			new_direction = (other.position - fish.position)
			attraction_orientation_direction += (new_direction / np.linalg.norm(new_direction))

		if distance <= Fish.ORIENTATION_DISTANCE:
			attraction_orientation_found = True
			attraction_orientation_direction += other.heading
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
	ATTRACTION_ALIGNMENT_WEIGHT = 0.5
	MAX_TURN = 0.1
	TURN_NOISE_SCALE = 0.1 # standard deviation in noise
	SPEED = 1.0
	DESIRED_DIRECTION = np.array([1, 0]) # Desired direction of informed fish is towards the right when [1, 0]
	DESRIRED_DIRECTION_WEIGHT = 0.8 # Weighting term is strength between swimming
									# towards desired direction and schooling (1 is all desired direction, 0 is all
									# schooling and ignoring desired ditrection

	def __init__(self, position, heading, informed=False):
		"""initial values for position and headning
		setting up the informed fish from a subset of NUM_FISHES
		pink fish are informed, but fish are naive"""
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

			noise = np.random.normal(0,Fish.TURN_NOISE_SCALE, World.NUM_FISHES) # adding noise to new_heading
			# new_heading += noise

			dot = np.dot(new_heading, self.heading)
			dot = min(1.0, dot)
			dot = max(-1.0, dot)
			angle_between = np.arccos(dot)
			if angle_between > Fish.MAX_TURN:
				new_heading = rotate_towards(self.heading, new_heading, Fish.MAX_TURN)

			self.heading = new_heading

def main():
	# initialize the world and all the fish
	world = World()

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