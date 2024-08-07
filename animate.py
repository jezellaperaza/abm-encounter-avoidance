import matplotlib.pyplot as plt
import matplotlib.animation

import simulation

TIME_FRAME = 10000

# specific for animation
def color(fish):
    if fish.collided_with_turbine_base:
        return "green"
    if fish.struck_by_turbine_blade:
        return "purple"
    else:
        return "blue"


def main():

    world = simulation.World()
    world.update()

    x, y, z = [], [], []
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, s=5)
    # ax.view_init(90, 0)
    sc._offsets3d = []
    for d in range(simulation.DIMENSIONS):
        sc._offsets3d.append([f.position[d] for f in world.fishes])


    xt, yt, zt = [], [], []
    turbine_scatter = ax.scatter(xt, yt, zt, s=simulation.TURBINE_BASE_RADIUS * 10)

    ax.set_xlim(0, simulation.WORLD_SIZE[0])
    ax.set_ylim(0, simulation.WORLD_SIZE[1])
    ax.set_zlim(0, simulation.WORLD_SIZE[2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # plt.show()

    def animate(_):

        world.update()

        # We were using circular boundary conditions so this should never happen.
        sc._offsets3d = []
        for d in range(simulation.DIMENSIONS):
            sc._offsets3d.append([f.position[d] for f in world.fishes])

        sc.set_color([f.color for f in world.fishes])

        turbines = [world.turbine_base, world.turbine_blade]
        turbine_scatter._offsets3d = []
        for d in range(simulation.DIMENSIONS):
            turbine_scatter._offsets3d.append([t.position[d] for t in turbines])

        turbine_scatter.set_color(["red", "green"])

        if all(f.left_environment for f in world.fishes):
            print("All fish have left the environment in frame", world.frame_number)
            ani.event_source.stop()

    ani = matplotlib.animation.FuncAnimation(fig, animate, frames=TIME_FRAME, interval=100, repeat=False)
    plt.show()

    world.print_close_out_message()


main()