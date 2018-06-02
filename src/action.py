import random


class Action:
    DO_NOTHING = 0
    MOVE_N = 1
    MOVE_NE = 2
    MOVE_E = 3
    MOVE_SE = 4
    MOVE_S = 5
    MOVE_SW = 6
    MOVE_W = 7
    MOVE_NW = 8
    TAKE = 9
    USE = 10
    DROP = 11

    NB_POSSIBLE_ACTIONS = 12
    NB_POSSIBLE_MOVE_ACTION = 9

    @staticmethod
    def random_action():
        return random.randint(0,Action.NB_POSSIBLE_ACTIONS-1)

    @staticmethod
    def to_str(action):
        if action == Action.DO_NOTHING:
            return 'do_nothing'
        elif action == Action.MOVE_N:
            return 'move N'
        elif action == Action.MOVE_NE:
            return 'move NE'
        elif action == Action.MOVE_E:
            return 'move E'
        elif action == Action.MOVE_SE:
            return 'move SE'
        elif action == Action.MOVE_S:
            return 'move S'
        elif action == Action.MOVE_SW:
            return 'move SW'
        elif action == Action.MOVE_W:
            return 'move W'
        elif action == Action.MOVE_NW:
            return 'move NW'
        elif action == Action.TAKE:
            return 'take'
        elif action == Action.USE:
            return 'use'
        elif action == Action.DROP:
            return 'drop'
        else:
            raise 'Action.to_str(): action not recognized'

    # @staticmethod
    # def random_move():
    #     return random.randint(0,Action.NB_POSSIBLE_MOVE_ACTION-1)

    to_dX = {
        DO_NOTHING : 0,
        MOVE_N : 0,
        MOVE_NE : 1,
        MOVE_E : 1,
        MOVE_SE : 1,
        MOVE_S : 0,
        MOVE_SW : -1,
        MOVE_W : -1,
        MOVE_NW : -1,
        TAKE : 0,
        USE : 0,
        DROP : 0
    }

    to_dY = {
        DO_NOTHING : 0,
        MOVE_N : 1,
        MOVE_NE : 1,
        MOVE_E : 0,
        MOVE_SE : -1,
        MOVE_S : -1,
        MOVE_SW : -1,
        MOVE_W : 0,
        MOVE_NW : 1,
        TAKE : 0,
        USE : 0,
        DROP : 0
    }

    def to_move(dx, dy):
        if dx is -1:
            if dy is -1:
                return Action.MOVE_SW
            elif dy is 0:
                return Action.MOVE_W
            else:
                return Action.MOVE_NW
        elif dx is 0:
            if dy is -1:
                return Action.MOVE_S
            elif dy is 0:
                return Action.DO_NOTHING
            else:
                return Action.MOVE_N
        else:
            if dy is -1:
                return Action.MOVE_SE
            elif dy is 0:
                return Action.MOVE_E
            else:
                return Action.MOVE_NE
