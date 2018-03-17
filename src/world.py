import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

import random
random.seed()

class GameEntity:
    def __init__(self):
        self.x = 0
        self.y = 0


class World(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        # self.action_space = spaces.Discrete(2)
        # self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None

        self.game_height = 40
        self.game_width = 60

        self.agent = GameEntity()

        # self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        :param: action: an array of -1, 0 or 1. [left_right, up_down].[1,-1] == right down.
        """
        # update new agent position
        self.agent.x +=  action[0] if self.agent.x + action[0] < self.game_width  and self.agent.x + action[0] >= 0 else 0
        self.agent.y +=  action[1] if self.agent.y + action[1] < self.game_height and self.agent.y + action[1] >= 0 else 0

        # do the rest... TODO
        return 0,0,0,{}
        # return np.array(self.state), reward, done, {}

    def reset(self):
        # init agent's position
        # self.agent.x = random.randint(0, self.game_width)
        # self.agent.y = random.randint(0, self.game_height)
        self.agent.x = self.game_width -1
        self.agent.y = self.game_height -1


        # do the rest TODO

        # return np.array(self.state)

    def render(self, mode='human', close=False):
        def add_entity_to_renderer(entity):
            entity.geom = rendering.make_circle(radius)
            entity.transform = rendering.Transform()
            entity.geom.add_attr(entity.transform)
            self.viewer.add_geom(entity.geom)

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

        render_entity(self.agent)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


if __name__ == "__main__":
    import time
    import pygame
    pygame.init()
    screen = pygame.display.set_mode([50,50]) # needed to capture when the keyboard is pressed (the focus has to be on this window)

    world = World()
    world = gym.wrappers.Monitor(world, 'video_output/', force=True) # force=True to overwrite the videos
    world.reset()
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

        world.step([x, y])

    for i in range(0,480):
        time.sleep(0.02)
        move_agent(world)
        world.render()
