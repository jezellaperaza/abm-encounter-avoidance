import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np

import simulation

def color(fish):
	if fish.collided_with_turbine:
		return "green"
	if fish.struck_by_turbine:
		return "purple"
	else:
		return "blue"



def main():
    np.random.seed(123)

    world = simulation.World()
    frame_number = 0


    x, y, z = [], [], []
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, s=5)
    # ax.view_init(azim=270, elev=0)

    ax.set_xlim(0, simulation.World.SIZE[0])
    ax.set_ylim(0, simulation.World.SIZE[1])
    ax.set_zlim(0, simulation.World.SIZE[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    headings = []

    def animate(_):


        world.update()


        # TODO - why were we using min(f.position[0], simulation.World.SIZE[0]) here?
        # We were using circular boundary conditions so this should never happen.
        # TODO - Jezella: what if we end up wanting to do periodic boundaries for the top and bottom of enivornment? Assuming where
        # TODO - min(f.position[0], simulation.World.SIZE[0]) might come from.
        sc._offsets3d = []
        for d in range(world.DIMENSIONS):
        	sc._offsets3d.append([f.position[d] for f in world.fishes])

        sc.set_color([f.color for f in world.fishes])

        if all(f.left_environment for f in world.fishes):
            print("All fish have left the environment in frame", world.frame_number)
            ani.event_source.stop()


    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=2, interval=100, repeat=True)
    plt.show()

    fish_in_zoi_count = len([f for f in world.fishes if f.in_zoi])
    fish_in_ent_count = len([f for f in world.fishes if f.in_entrainment])
    fish_collided_count = len([f for f in world.fishes if f.collided_with_turbine])
    fish_struck_count = len([f for f in world.fishes if f.struck_by_turbine])

    print("Number of fish in ZOI:", fish_in_zoi_count)
    print("Number of fish in entrainment:", fish_in_ent_count)
    print("Number of fish collided with the turbine:", fish_collided_count)
    print("Number of fish struck by the turbine:", fish_struck_count)

main()