import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from action import *

class Direction:
    N = 11
    NE = 12
    E = 13
    SE = 14
    S = 15
    SW = 16
    W = 17
    NW = 18

    dx = {
        N: 0,
        NE: 1,
        E: 1,
        SE: 1,
        S: 0,
        SW: -1,
        W: -1,
        NW: -1
    }

    dy = {
        N: 1,
        NE: 1,
        E: 0,
        SE: -1,
        S: -1,
        SW: -1,
        W: 0,
        NW: 1
    }

    inverse_x_direction = {
        N: N,
        NE: NW,
        E: W,
        SE: SW,
        S: S,
        SW: SE,
        W: E,
        NW: NE
    }

    inverse_y_direction = {
        N: S,
        NE: SE,
        E: E,
        SE: NE,
        S: N,
        SW: NW,
        W: W,
        NW: SW
    }

import random
random.seed()
random_dir = [Direction.N, Direction.NE, Direction.E, Direction.SE, Direction.SW, Direction.W, Direction.NW]
random.shuffle(random_dir)

class GameEntity:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Ennemy(GameEntity):

    def __init__(self, x=0, y=0, direction=None):
        super().__init__(x, y)
        self.direction = direction if direction is not None else random_dir.pop(random.randint(0,len(random_dir)-1))
        print(self.direction)

    def move(self, world):
        self.x += Direction.dx[self.direction]
        self.y += Direction.dy[self.direction]

        if self.x <= 0 or self.x >= world.game_width:
            self.direction = Direction.inverse_x_direction[self.direction]
        if self.y <= 0 or self.y >= world.game_height:
            self.direction = Direction.inverse_y_direction[self.direction]


class PursuingEnnemy(Ennemy):
    def __init__(self, x=0, y=0, direction=None):
        super().__init__(x,y,direction)

    def move(self, world):
        agent = world.agent

        # update direction
        if agent.y > self.y:
            if agent.x > self.x:
                self.direction = Direction.NE
            elif agent.x < self.x:
                self.direction = Direction.NW
            else:
                self.direction = Direction.N
        elif agent.y < self.y:
            if agent.x > self.x:
                self.direction = Direction.SE
            elif agent.x < self.x:
                self.direction = Direction.SW
            else:
                self.direction = Direction.S
        else:
            if agent.x > self.x:
                self.direction = Direction.E
            elif agent.x < self.x:
                self.direction = Direction.W

        self.x += 0.5 * Direction.dx[self.direction]
        self.y += 0.5 * Direction.dy[self.direction]

CONFIG = {
    'ennemies' : True,
}

class World(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, configuration=CONFIG):
        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(-high, high)
        self.config = configuration

        self.seed()
        self.viewer = None
        self.state = None

        self.game_height = 40
        self.game_width = 60

        self.agent = GameEntity()
        if self.config['ennemies']:
            self.ennemies = [Ennemy(25, 25), Ennemy(50,12), Ennemy(55,35), PursuingEnnemy(3,50)]

        self.game_over = True

        # self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        :param: action: an array of -1, 0 or 1. [left_right, up_down].[1,-1] == right down.
        """
        reward = 1

        # update new agent position
        self.agent.x +=  Action.to_dX[action] if self.agent.x + Action.to_dX[action] < self.game_width  and self.agent.x + Action.to_dX[action] >= 0 else 0
        self.agent.y +=  Action.to_dY[action] if self.agent.y + Action.to_dY[action] < self.game_height and self.agent.y + Action.to_dY[action] >= 0 else 0

        if self.config['ennemies']:
            self._manage_enemies()

        # do the rest... TODO
        return 0,reward,self.game_over,{}
        # return np.array(self.state), reward, done, {}

    def _manage_enemies(self):
        # update ennemies position
        for ennemy in self.ennemies:
            ennemy.move(self)

            # check if game is finished
            if self.distance(self.agent, ennemy) <= 1:
                self.game_over = True


    def distance(self, entity1, entity2):
       return ( (entity1.x-entity2.x)**2 + (entity1.y-entity2.y)**2 ) ** 0.5

    def reset(self):
        # init agent's position
        # self.agent.x = random.randint(0, self.game_width)
        # self.agent.y = random.randint(0, self.game_height)
        self.agent.x = 30
        self.agent.y = 0
        self.game_over = False


        # do the rest TODO

        # return np.array(self.state)

    def render(self, mode='human', close=False):
        def add_entity_to_renderer(entity):
            entity.geom = rendering.make_circle(radius)
            entity.transform = rendering.Transform()
            entity.geom.add_attr(entity.transform)
            self.viewer.add_geom(entity.geom)
            if type(entity) == GameEntity:
                entity.geom.set_color(0,0,128)
            elif isinstance(entity, Ennemy):
                entity.geom.set_color(128,0,0)

        def render_entity(entity):
            entity.transform.set_translation(entity.x*scale_x+radius, entity.y*scale_y+radius)

        screen_width = 600
        screen_height = 400
        radius = 10
        scale_x = (screen_width - 1 * radius) / self.game_width
        scale_y = (screen_height - 1 * radius) / self.game_height

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            add_entity_to_renderer(self.agent)

            if self.config['ennemies']:
                for ennemy in self.ennemies:
                    add_entity_to_renderer(ennemy)

        render_entity(self.agent)
        if self.config['ennemies']:
            for ennemy in self.ennemies:
                render_entity(ennemy)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


if __name__ == "__main__":
    import time
    import pygame
    pygame.init()
    screen = pygame.display.set_mode([50,50]) # needed to capture when the keyboard is pressed (the focus has to be on this window)

    CONFIG['ennemies'] = False
    world = World()
    world = gym.wrappers.Monitor(world, 'video_output/', force=True) # force=True to overwrite the videos
    world.reset()
    game_over = False
    world.render()

    def move_agent(world):
        x = 0
        y = 0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            x -= 1
        if keys[pygame.K_RIGHT]:
            x += 1
        if keys[pygame.K_UP]:
            y += 1
        if keys[pygame.K_DOWN]:
            y -= 1
        pygame.event.pump()

        _, _, game_over, _ = world.step(Action.to_move(x, y))
        return game_over

    while not game_over:
        time.sleep(0.02)
        game_over = move_agent(world)
        world.render()
