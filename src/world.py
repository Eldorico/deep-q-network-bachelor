import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

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

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        pass
        # return np.array(self.state), reward, done, {}

    def reset(self):
        pass
        # return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        # world_width = self.x_threshold*2
        # scale = screen_width/world_width
        # carty = 100 # TOP OF CART
        # polewidth = 10.0
        # polelen = scale * 1.0
        # cartwidth = 50.0
        # cartheight = 30.0

        if self.viewer is None:
            # https://github.com/openai/gym/blob/master/gym/envs/classic_control/rendering.py
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            agent = rendering.make_circle()
            self.agentTrans = rendering.Transform()
            agent.add_attr(self.agentTrans)
            self.viewer.add_geom(agent)
            # l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            # axleoffset =cartheight/4.0
            # cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # self.carttrans = rendering.Transform()
            # cart.add_attr(self.carttrans)
            # self.viewer.add_geom(cart)
            # l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            # pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # pole.set_color(.8,.6,.4)
            # self.poletrans = rendering.Transform(translation=(0, axleoffset))
            # pole.add_attr(self.poletrans)
            # pole.add_attr(self.carttrans)
            # self.viewer.add_geom(pole)
            # self.axle = rendering.make_circle(polewidth/2)
            # self.axle.add_attr(self.poletrans)
            # self.axle.add_attr(self.carttrans)
            # self.axle.set_color(.5,.5,.8)
            # self.viewer.add_geom(self.axle)
            # self.track = rendering.Line((0,carty), (screen_width,carty))
            # self.track.set_color(0,0,0)
            # self.viewer.add_geom(self.track)

        # if self.state is None: return None
        #
        # x = self.state
        # cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        # self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()


if __name__ == "__main__":
    world = World()
    world.render()
    while True:
        pass
