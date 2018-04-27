import matplotlib.pyplot as plt
import numpy as np

from world import *

# create the world
def reward_function(world):
    if world.game_over:
        return - 5
    else:
        max_distance = 10
        security_distance = 5
        smallest_distance_ennemy_collision_course = float('Inf')
        for ennemy in world.ennemies:
            if Direction.is_in_collision_course(ennemy, world.agent, security_distance):
                distance = Direction.distance(ennemy, world.agent)
                if distance < smallest_distance_ennemy_collision_course:
                    smallest_distance_ennemy_collision_course = distance
        if smallest_distance_ennemy_collision_course >= max_distance:
            return 1
        elif smallest_distance_ennemy_collision_course <=2:
            return 0.01 * smallest_distance_ennemy_collision_course
        else:
            return ((smallest_distance_ennemy_collision_course -2) /max_distance) ** 0.4
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
Z = np.zeros((world.game_width, world.game_height))
for x in X:
    for y in Y:
        world.agent.x, world.agent.y = x, y
        Z[world.game_height - 1 - y][x] = reward_function(world)

# add the ennemies
for ennemy in world.ennemies:
    ennemy_y = int(world.game_height - 1 - ennemy.y)
    ennemy_x = int(ennemy.x)
    Z[ennemy_y][ennemy_x] = -0.5

    # print("%f, %f" %(ennemy.x, ennemy.y))
    # print(Direction.toStr[ennemy.direction])

    start_x_arrow = ennemy_x + Direction.dx[ennemy.direction]
    start_y_arrow = ennemy_y - Direction.dy[ennemy.direction]
    end_x_arrow = ennemy_x - Direction.dx[ennemy.direction]
    end_y_arrow = ennemy_y + Direction.dy[ennemy.direction]
    plt.annotate('', xy=(start_x_arrow,start_y_arrow), xytext=(end_x_arrow, end_y_arrow),
                    arrowprops=dict(facecolor='red', shrink=0.05))

pos = plt.imshow(Z, cmap='Greens', interpolation='none')
plt.colorbar(pos)
plt.show()
