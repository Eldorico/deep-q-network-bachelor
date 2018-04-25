import numpy as np
import math

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

    @staticmethod
    def distance(entity1, entity2):
       return ( (entity1.x-entity2.x)**2 + (entity1.y-entity2.y)**2 ) ** 0.5

    @staticmethod
    def get_direction_to(me, target):
        if target.y > me.y:
            if target.x > me.x:
                return Direction.NE
            elif target.x < me.x:
                return Direction.NW
            else:
                return Direction.N
        elif target.y < me.y:
            if target.x > me.x:
                return Direction.SE
            elif target.x < me.x:
                return Direction.SW
            else:
                return Direction.S
        else:
            if target.x > me.x:
                return Direction.E
            elif target.x < me.x:
                return Direction.W

    @staticmethod
    def is_in_direction(ennemy, target):
        """ if the ennemy on a (more or less) direction towards the agent
        """
        distance_before_ennemy_move = Direction.distance(ennemy, target)
        ennemy.x += Direction.dx[ennemy.direction]
        ennemy.y += Direction.dy[ennemy.direction]
        distance_after_ennemy_move = Direction.distance(ennemy, target)

        if distance_after_ennemy_move <= distance_before_ennemy_move:
            return True
        return False

    @staticmethod
    def is_in_collision_course(ennemy, target, min_dist_from_course):
        if not Direction.is_in_direction(ennemy, target):
            return False

        # get cartesian equation of a line
        a = Direction.dy[ennemy.direction]
        b = -Direction.dx[ennemy.direction]
        c = -( a * ennemy.x + b * ennemy.y )

        # get agent distance from the line of the agent
        numerator = abs(a*target.x + b*target.y + c)
        denominator = math.sqrt(a**2 + b**2)
        agent_distance_from_line = numerator / denominator

        # return the result
        if agent_distance_from_line < min_dist_from_course:
            return True
        return False


class State:
    AGENT = 10  # this must not overlap the Direction

    ENNEMY_AGENT_STD_VALUE = {
        Direction.N: -0.5,
        Direction.NE: -0.4,
        Direction.E: -0.3,
        Direction.SE: -0.2,
        Direction.S: -0.1,
        Direction.SW: +0.1,
        Direction.W: 0.2,
        Direction.NW: 0.3,
        AGENT:  0.4
    }

    def __init__(self, world_size_x, world_size_y):
        self.world_width = world_size_x
        self.world_height = world_size_y
        self.ennemy_agent_positions = np.zeros(world_size_x*world_size_y)
        self.object_positions = np.zeros(world_size_x*world_size_y)
        self.agent_state = np.ones(2)

    def place_ennemy(self, ennemy):
        self.ennemy_agent_positions[self.to_1D(ennemy.x, ennemy.y)] = State.ENNEMY_AGENT_STD_VALUE[ennemy.direction]

    def place_agent(self, pos_x, pos_y):
        self.ennemy_agent_positions[self.to_1D(pos_x, pos_y)] = State.ENNEMY_AGENT_STD_VALUE[State.AGENT]

    def to_1D(self, x, y):
        result = int(y * self.world_width + x)
        return result if result >= 0 and result < self.ennemy_agent_positions.size else self.ennemy_agent_positions.size -1

    def agent_ennemies_pos_layer_to_string(self):
        return str(np.reshape(self.ennemy_agent_positions, (self.world_height, self.world_width)))

    @staticmethod
    def get_ennemy_agent_layer_shape(world):
        return world.game_width * world.game_height

    def get_ennemy_agent_layer_only(self):
        return self.ennemy_agent_positions.reshape(1,len(self.ennemy_agent_positions))
