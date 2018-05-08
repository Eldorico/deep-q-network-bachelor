import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from action import *
from state import *

import random
import time
seed = time.time()-round(time.time())
random.seed(seed)
print("random seed: %f" % seed)
# random_dir = [Direction.N, Direction.NE, Direction.E, Direction.SE, Direction.S, Direction.SW, Direction.W, Direction.NW]
# random.shuffle(random_dir)

class GameEntity:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

class Ennemy(GameEntity):

    def __init__(self, position, direction=None):
        super().__init__(position[0], position[1])
        self.direction = direction if direction is not None else random.randint(1,8) # _dir.pop(random.randint(0,len(random_dir)-1))
        # print(self.direction)

    def move(self, world):
        self.x += Direction.dx[self.direction]
        self.y += Direction.dy[self.direction]

        if self.x <= 0 or self.x >= world.game_width:
            self.direction = Direction.inverse_x_direction[self.direction]
        if self.y <= 0 or self.y >= world.game_height:
            self.direction = Direction.inverse_y_direction[self.direction]


class PursuingEnnemy(Ennemy):
    def __init__(self, position, direction=None):
        super().__init__(position, direction)

    def move(self, world):
        agent = world.agent

        # update direction
        self.direction = Direction.get_direction_to(self, agent)

        self.x += 0.5 * Direction.dx[self.direction]
        self.y += 0.5 * Direction.dy[self.direction]


class World(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, configuration):
        if  'reward_function' not in configuration:
            def r_function(world):
                return 1
            configuration['reward_function'] = r_function
        if 'print_reward' not in configuration:
            configuration['print_reward'] = False
        if 'render' not in configuration:
            configuration['render'] = False

        self.config = configuration

        self.seed()
        self.viewer = None
        self.state = None
        self.game_over = True
        self.score = 0
        self.total_reward = 0
        self.reward_function = configuration['reward_function']

        # self.game_width =  60
        # self.game_height = 40
        self.game_width =  10
        self.game_height = 10

        self.agent = GameEntity()
        self.ennemies = []


        # self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        :param: action: an array of -1, 0 or 1. [left_right, up_down].[1,-1] == right down.
        """
        world_debug_info = {'agent_x': self.agent.x, 'agent_y': self.agent.y}
        if self.config['ennemies']:
            world_debug_info['ennemies_position'] = [(e.x, e.y, e.direction) for e in self.ennemies]

        current_state = State(self.game_width, self.game_height)

        if action < 0 or action >= Action.NB_POSSIBLE_ACTIONS:
             sys.stderr.write("World.step(): action not in action space: %d \n" % action)
             exit(-1)

        # update new agent position
        self.agent.x +=  Action.to_dX[action] if self.agent.x + Action.to_dX[action] < self.game_width  and self.agent.x + Action.to_dX[action] >= 0 else 0
        self.agent.y +=  Action.to_dY[action] if self.agent.y + Action.to_dY[action] < self.game_height and self.agent.y + Action.to_dY[action] >= 0 else 0
        current_state.place_agent(self.agent.x,self.agent.y)

        if self.config['ennemies']:
            self._manage_enemies(current_state)

        self.score += 1

        # set the reward
        reward = self.reward_function(self)
        self.total_reward += reward
        world_debug_info['total_reward'] = self.total_reward

        if self.config['print_reward']:
            print("reward: %f - distance: %f" % (reward, smallest_distance_ennemy_collision_course) )
        # reward = 1

        # render if we have to render the game (show the world in a window)
        if self.config['render']:
            self.render()

        # do the rest... TODO
        world_debug_info['score'] = self.score
        return current_state, reward, self.game_over, world_debug_info
        # return np.array(self.state), reward, done, {}

    def _manage_enemies(self, current_state):
        # update ennemies position
        for ennemy in self.ennemies:
            ennemy.move(self)
            current_state.place_ennemy(ennemy)

            # check if game is finished
            if Direction.distance(self.agent, ennemy) <= 2:
                self.game_over = True

    def reset(self):
        # init agent's position
        # self.agent.x = random.randint(0, self.game_width)
        # self.agent.y = random.randint(0, self.game_height)
        # self.agent.x = 30
        # self.agent.y = 0
        self.game_over = False

        # reset the window renderer
        if self.config['render']:
            if self.viewer is not None:
                self.viewer.close()
            self.viewer = None

        if self.config['ennemies']:
            self.ennemies = [PursuingEnnemy(self.rand_pos())]
            # self.ennemies = [Ennemy(self.rand_pos()), Ennemy(self.rand_pos()), Ennemy(self.rand_pos()), PursuingEnnemy(self.rand_pos())]

        # do the rest TODO

        self.choose_random_but_safe_start_location_for_agent()

        current_state, _, _, _ = self.step(Action.DO_NOTHING)
        self.score = 0
        self.total_reward = 0

        return current_state

        # return np.array(self.state)

    def rand_pos(self):
        return random.randint(0, self.game_width-1), random.randint(0, self.game_height-1)

    def choose_random_but_safe_start_location_for_agent(self):
        agent_too_close_from_ennemies = True
        while agent_too_close_from_ennemies:
            self.agent.x = random.randint(0, self.game_width)
            self.agent.y = random.randint(0, self.game_height)
            agent_too_close_from_ennemies = False
            if self.config['ennemies']:
                for ennemy in self.ennemies:
                    if Direction.distance(self.agent, ennemy) < 3:
                        agent_too_close_from_ennemies = True
                        break

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

        # start
        # screen_width = 600
        # screen_height = 400
        screen_width = 100
        screen_height = 100
        radius = 10
        scale_x = (screen_width - 1 * radius) / self.game_width
        scale_y = (screen_height - 1 * radius) / self.game_height

        # create renderer if no renderer has been created
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            add_entity_to_renderer(self.agent)

            if self.config['ennemies']:
                for ennemy in self.ennemies:
                    add_entity_to_renderer(ennemy)

        # render agent + ennemies
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

    def default_reward(world):
        return 1
    CONFIG = {
        'ennemies' : True,
        'print_reward' : False,
        'reward_function': default_reward,
        'render' : True
    }
    world = World(CONFIG)
    world = gym.wrappers.Monitor(world, 'video_output/', force=True) # force=True to overwrite the videos
    world.reset()
    game_over = False

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

        state, _, game_over, debug = world.step(Action.to_move(x, y))
        return state, game_over, debug

    time.sleep(5) # to have time to place the windows and start to play
    while not game_over:
        # time.sleep(0.02)
        time.sleep(0.5)
        state, game_over, debug = move_agent(world)

    print("Score: %d" % debug['score'])
