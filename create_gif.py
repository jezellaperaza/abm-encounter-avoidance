from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import shutil

import simulation

TIME_FRAME = 1000


def color(fish):
    if fish.collided_with_turbine:
        return "green"
    if fish.struck_by_turbine:
        return "purple"
    else:
        return "blue"


def main():
    parent_dir = '/Users/jezellaperaza/Documents/GitHub'
    num_simulations = 1

    def animate():

        world.update()

        world.all_fish_left = all(f.left_environment for f in world.fishes)
        if world.all_fish_left:
            print("All fish have left the environment in frame", world.frame_number)

    for sim_num in range(num_simulations):
        # initialize the world and all the fish
        world = simulation.World()
        world.update()

        try:
            os.mkdir(os.path.join(parent_dir, str(sim_num)))
        except:
            shutil.rmtree(os.path.join(parent_dir, str(sim_num)))
            os.mkdir(os.path.join(parent_dir, str(sim_num)))

        # Run the fish animation loop for the current simulation
        continue_simulation = True

        while continue_simulation and world.frame_number < TIME_FRAME:

            x, y, z = [], [], []
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            sc = ax.scatter(x, y, z, s=5)
            # ax.view_init(0, 0)
            sc._offsets3d = []
            for d in range(simulation.DIMENSIONS):
                sc._offsets3d.append([f.position[d] for f in world.fishes])

            xt, yt, zt = [], [], []
            turbine_scatter = ax.scatter(xt, yt, zt, s=simulation.TURBINE_RADIUS * 20)

            ax.set_xlim(0, simulation.WORLD_SIZE[0])
            ax.set_ylim(0, simulation.WORLD_SIZE[1])
            ax.set_zlim(0, simulation.WORLD_SIZE[2])

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            turbines = [world.turbine_base, world.turbine_blade]
            turbine_scatter._offsets3d = []
            for d in range(simulation.DIMENSIONS):
                turbine_scatter._offsets3d.append([t.position[d] for t in turbines])

            turbine_scatter.set_color(["red", "green"])

            animate()

            # When you save the figure:
            plt.savefig(f'{parent_dir}/{sim_num}/{world.frame_number}.png',
                        transparent=False,
                        facecolor='white')
            plt.close()

            if all(f.left_environment for f in world.fishes) or world.frame_number >= TIME_FRAME:
                continue_simulation = False

        world.print_close_out_message()

        # Create the gif when each simulation ends
        images = []

        filenames = os.listdir(os.path.join(parent_dir, str(sim_num)))
        filenames = np.array(filenames)
        nums = np.array([int(im.split('/')[-1].split('.')[0]) for im in
                         filenames])  # pull image num from filepaths and convert to ints
        sort_i = np.argsort(nums)  # get indices of sorted order

        for filename in filenames[sort_i]:
            images.append(imageio.v2.imread(os.path.join(parent_dir, str(sim_num), filename)))

        fps = 1
        imageio.mimsave(f'{parent_dir}/sim_{sim_num}.gif', images, duration=world.frame_number / fps, loop=1)


main()
