from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import math
import imageio
import os
import shutil

import simulation

TIME_FRAME = 100


def color(fish):
	if fish.collided_with_turbine:
		return "green"
	if fish.struck_by_turbine:
		return "purple"
	else:
		return "blue"


def main():
    parent_dir = 'C:/Users/JPeraza/Documents/UW Winter Quarter 2024/Sample'
    num_simulations = 1

    def animate():


        world.update()

        x = [min(f.position[0], World.SIZE[0]) for f in world.fishes]
        y = [min(f.position[1], World.SIZE[1]) for f in world.fishes]
        z = [min(f.position[2], World.SIZE[2]) for f in world.fishes]

        sc._offsets3d = (x, y, z)

        colors = [color(f) for f in world.fishes]
        sc.set_color(colors)

        world.all_fish_left = all(f.left_environment for f in world.fishes)
        if world.all_fish_left:
            print("All fish have left the environment in frame", world.frame_number)


    for sim_num in range(num_simulations):
        # initialize the world and all the fish
        world = simulation.World()


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
            # ax.view_init(azim=270, elev=10)

            ax.set_xlim(0, World.SIZE[0])
            ax.set_ylim(0, World.SIZE[1])
            ax.set_zlim(0, World.SIZE[2])

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

            animate()

            # When you save the figure:
            plt.savefig(f'{parent_dir}/{sim_num}/{frame_number}.png',
                        transparent=False,
                        facecolor='white')
            plt.close()

            if all(f.left_environment for f in world.fishes) or frame_number >= World.TIME_FRAME:
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

        fps = 10 # goal to make this 1? or will we end up leaving it at 10?
        imageio.mimsave(f'{parent_dir}/sim_{sim_num}.gif', images, duration=frame_number / fps, loop=1)

main()
