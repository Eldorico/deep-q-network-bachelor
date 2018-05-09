import matplotlib.pyplot as plt
import numpy as np

from world import *

""" put the reward fuunction here
#############################################################################"""
def reward_function(world):
    if world.game_over:
        return - 1
    else:
        safe_distance = 6
        min_distance = float('inf')
        for ennemy in world.ennemies:
            distance = Direction.distance(ennemy, world.agent)
            if distance < min_distance:
                min_distance = distance

        if min_distance >= safe_distance:
            return 1
        elif min_distance <= 1:
            return -1
        else:
            return math.log(min_distance+0.01) -1
"""#############################################################################
"""

# create the world
world = World({
    'ennemies' : True,
    'print_reward' : False,
    'reward_function' : reward_function
})
world.reset()

# debug
# world.ennemies[0].direction = Direction.NW
# world.ennemies[0].x = 2
# world.ennemies[0].y = 4
# world.agent.x = 1
# world.agent.y = 4
# print(reward_function(world))


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

    # print("%f, %f" %(ennemy.x, ennemy.y))
    # print(Direction.toStr[ennemy.direction])

    # start_x_arrow = ennemy_x + Direction.dx[ennemy.direction]
    # start_y_arrow = ennemy_y - Direction.dy[ennemy.direction]
    # end_x_arrow = ennemy_x - Direction.dx[ennemy.direction]
    # end_y_arrow = ennemy_y + Direction.dy[ennemy.direction]
    # plt.annotate('', xy=(start_x_arrow,start_y_arrow), xytext=(end_x_arrow, end_y_arrow),
    #                 arrowprops=dict(facecolor='red', shrink=0.05))

pos = plt.imshow(Z, cmap='Greens', interpolation='none')
plt.colorbar(pos)
plt.show()
