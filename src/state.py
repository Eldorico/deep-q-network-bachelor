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
    NO_DIRECTION = -1

    toStr = {
        N : 'N',
        NE : 'NE',
        E : 'E',
        SE : 'SE',
        S : 'S',
        SW : 'SW',
        W : 'W',
        NW : 'NW',
        NO_DIRECTION: '-'
    }


    dx = {
        N: 0,
        NE: 1,
        E: 1,
        SE: 1,
        S: 0,
        SW: -1,
        W: -1,
        NW: -1,
        NO_DIRECTION: 0
    }

    dy = {
        N: 1,
        NE: 1,
        E: 0,
        SE: -1,
        S: -1,
        SW: -1,
        W: 0,
        NW: 1,
        NO_DIRECTION: 0
    }

    inverse_x_direction = {
        N: N,
        NE: NW,
        E: W,
        SE: SW,
        S: S,
        SW: SE,
        W: E,
        NW: NE,
        NO_DIRECTION: 0
    }

    inverse_y_direction = {
        N: S,
        NE: SE,
        E: E,
        SE: NE,
        S: N,
        SW: NW,
        W: W,
        NW: SW,
        NO_DIRECTION: 0
    }

    @staticmethod
    def distance(entity1, entity2):
       return ( (entity1.x-entity2.x)**2 + (entity1.y-entity2.y)**2 ) ** 0.5

    @staticmethod
    def distance2(x1, y1, x2, y2):
        return ( (x1-x2)**2 + (y1-y2)**2 ) ** 0.5

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
            else:
                return Direction.NO_DIRECTION

    @staticmethod
    def is_in_direction(ennemy, target):
        """ if the ennemy on a (more or less) direction towards the agent
        """
        save_x, save_y = ennemy.x, ennemy.y

        distance_before_ennemy_move = Direction.distance(ennemy, target)
        ennemy.x += Direction.dx[ennemy.direction]
        ennemy.y += Direction.dy[ennemy.direction]
        distance_after_ennemy_move = Direction.distance(ennemy, target)

        ennemy.x, ennemy.y = save_x, save_y

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
    AGENT = 1  # this must not overlap the Direction
    ENNEMY = -1

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

    def __init__(self, world):
        self.world_width = world.game_width
        self.world_height = world.game_height
        self.ennemy_agent_positions = np.zeros(State.get_ennemy_agent_layer_shape(world))
        self.food_state = np.zeros(3) # stamina, food_x, food_y

    def place_ennemy(self, ennemy, i):
        self.ennemy_agent_positions[i*2+2] = ennemy.x / self.world_width
        self.ennemy_agent_positions[i*2+3] = ennemy.y / self.world_height

    def place_agent(self, pos_x, pos_y):
        self.ennemy_agent_positions[0] = pos_x / self.world_width
        self.ennemy_agent_positions[1] = pos_y / self.world_height

    def set_food_state(self, stamina, food_x_pos, food_y_pos):
        self.food_state[0] = stamina / 100.0
        self.food_state[1] = food_x_pos / self.world_width
        self.food_state[2] = food_y_pos / self.world_height

    def to_1D(self, x, y):
        result = int(y * self.world_width + x)
        return result if result >= 0 and result < self.ennemy_agent_positions.size else self.ennemy_agent_positions.size -1

    def agent_ennemies_pos_layer_to_string(self):
        return str(np.reshape(self.ennemy_agent_positions, (self.world_height, self.world_width)))

    @staticmethod
    def get_ennemy_agent_layer_shape(world):
        return 2 + 2 * len(world.ennemies)

    def get_ennemy_agent_layer_only(self):
        return self.ennemy_agent_positions.reshape(1,len(self.ennemy_agent_positions))

    def get_agent_position_layer(self):
        return self.ennemy_agent_positions[:2]

    def get_food_position_layer(self):
        return self.food_state[1:]

    def get_stamina_value(self):
        return self.food_state[0]

    def get_food_position_and_stamina_value(self):
        return self.food_state

    def get_min_distance_between_agent_ennemy(self):
        agent_x, agent_y = self.ennemy_agent_positions[0], self.ennemy_agent_positions[1]
        min_distance = float('inf')
        for i in range(2, len(self.ennemy_agent_positions), 2):
            ennemy_x, ennemy_y = self.ennemy_agent_positions[i], self.ennemy_agent_positions[i+1]
            min_distance = min(min_distance, Direction.distance2(agent_x, agent_y, ennemy_x, ennemy_y))
        return min_distance

    def get_distance_from_food(self):
        return Direction.distance2(self.ennemy_agent_positions[0], self.ennemy_agent_positions[1], self.food_state[1], self.food_state[2])
