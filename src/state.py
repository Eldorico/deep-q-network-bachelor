import numpy as np

class Direction:
    N = 1
    NE = 2
    E = 3
    SE = 4
    S = 5
    SW = 6
    W = 7
    NW = 8

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

class State:
    AGENT = 10  # this must not overlap the Direction

    def __init__(self, world_size_x, world_size_y):
        self.world_width = world_size_x
        self.world_height = world_size_y
        self.ennemy_agent_positions = np.zeros(world_size_x*world_size_y)
        self.object_positions = np.zeros(world_size_x*world_size_y)
        self.agent_state = np.ones(2)

    def place_ennemy(self, ennemy):
        self.ennemy_agent_positions[self.to_1D(ennemy.x, ennemy.y)] = ennemy.direction

    def place_agent(self, pos_x, pos_y):
        self.ennemy_agent_positions[self.to_1D(pos_x, pos_y)] = State.AGENT

    def to_1D(self, x, y):
        result = int(y * self.world_width + x)
        return result if result >= 0 and result < self.ennemy_agent_positions.size else self.ennemy_agent_positions.size -1

    def agent_ennemies_pos_layer_to_string(self):
        return str(np.reshape(self.ennemy_agent_positions, (self.world_height, self.world_width)))
