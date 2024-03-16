import pytest
import numpy as np

import simulation

class TestInsideCylinder():
	def test_base(self):
		assert simulation.inside_cylinder(np.array([0, 0, 0]), 0, 0, np.array([0, 0, 0]))

	def test_above(self):
		assert simulation.inside_cylinder(np.array([0, 0, 0]), 0, 10, np.array([0, 0, 8]))

	def test_too_far_above(self):
		assert not simulation.inside_cylinder(np.array([0, 0, 0]), 0, 10, np.array([0, 0, 11]))

	def test_next_to(self):
		assert simulation.inside_cylinder(np.array([10, 10, 0]), 5, 5, np.array([15, 10, 5]))

	def test_too_far(self):
		assert not simulation.inside_cylinder(np.array([10, 10, 0]), 5, 5, np.array([15, 10, 6]))


def test_sanity():
	"initialize the world and run one update to see if it blows up"
	world = simulation.World()
	world.update()