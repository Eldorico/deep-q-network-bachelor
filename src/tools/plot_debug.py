import matplotlib.pyplot as plt
import numpy as np

import os, sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from world import *

""" put the reward fuunction here
#############################################################################"""
def reward_function(world):
    if world.game_over:
        return - 5
    max_distance = 73
    return (1 - Direction.distance(world.agent, world.food) / max_distance)
    # if world.game_over:
    #     return - 1
    # else:
    #     safe_distance = 20
    #     min_distance = float('inf')
    #     for ennemy in world.ennemies:
    #         distance = Direction.distance(ennemy, world.agent)
    #         if distance < min_distance:
    #             min_distance = distance
    #
    #     if min_distance >= safe_distance:
    #         return math.log(safe_distance+0.01) -1
    #     elif min_distance <= 1:
    #         return -1
    #     else:
    #         return math.log(min_distance+0.01) -1
"""#############################################################################
"""

# create the world
world = World({
    'food' : True,
    # 'ennemies' : True,
    'print_reward' : False,
    'reward_function' : reward_function
})
world.reset()

# fill the Z with the rewards
X, Y = [i for i in range(world.game_width)], [i for i in range(world.game_height)] # np.mgrid[:world.game_width, :world.game_height]
Z = np.zeros((world.game_height, world.game_width))
for x in X:
    for y in Y:
        world.agent.x, world.agent.y = x, y
        Z[y][x-1] = reward_function(world)


# add the ennemies
for ennemy in world.ennemies:
    ennemy_y = int(ennemy.y)
    ennemy_x = int(ennemy.x-1)
    Z[ennemy_y][ennemy_x] = -0.5

pos = plt.imshow(Z, cmap='Greens', interpolation='none')
plt.colorbar(pos)
plt.show()
