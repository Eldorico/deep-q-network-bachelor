import os
import sys
sys.path.append('../src')
import numpy as np
import unittest

from network import *
from world import *
from state import *
from agent import *



class WorldTest(unittest.TestCase):
    def create_default_test_world(self):
        # create the world
        world_config = {
            'ennemies' : True
        }
        world = World(world_config)
        start_state = world.reset()
        return world, start_state

    def test_states_values_are_between_0_and_1(self):
        world, start_state = self.create_default_test_world()

        self.assertTrue(all([value >= -1 and value <1 for value in start_state.ennemy_agent_positions]))

        next_state, reward, game_over, world_informations = world.step(Action.random_action())

        self.assertEqual(world.score, 1)
        self.assertTrue(all([value >= -1 and value <1 for value in next_state.ennemy_agent_positions]))

    def test_agent_has_its_position_on_correct_input(self):
        world, start_state = self.create_default_test_world()

        nb_agent_position_count = 0
        for slot in start_state.get_ennemy_agent_layer_only()[0]:
            if slot == State.ENNEMY_AGENT_STD_VALUE[State.AGENT]:
                nb_agent_position_count += 1
        self.assertEqual(1, nb_agent_position_count)

        next_state, reward, game_over, world_informations = world.step(Action.random_action())

        nb_agent_position_count = 0
        for slot in next_state.get_ennemy_agent_layer_only()[0]:
            if slot == State.ENNEMY_AGENT_STD_VALUE[State.AGENT]:
                nb_agent_position_count += 1
        self.assertEqual(1, nb_agent_position_count)
