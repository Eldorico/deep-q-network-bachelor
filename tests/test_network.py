import os
import sys
sys.path.append('../src')

from network import *
from world import *
from state import *
import numpy as np
import unittest


class NetworkTest(unittest.TestCase):

    def create_session_and_world_and_simple_network(self):
        # create the world
        world_config = {
            'ennemies' : True
        }
        world = World(world_config)

        # create the session
        session = tf.Session()

        # create the neural network that will learn to avoid ennemies
        model = Model(session, 'model_test', State.get_ennemy_agent_layer_shape(world), 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [Action.NB_POSSIBLE_ACTIONS, 'linear']]
        )
        def imput_adapter(bus, next_state=False):
            if next_state:
                return bus['next_state'].get_ennemy_agent_layer_only()
            else:
                return bus['state'].get_ennemy_agent_layer_only()
        network = Network(
            model,
            imput_adapter,
            True,
            True
        )

        return session, world, network

    def test_predict(self):
        session, world, network = self.create_session_and_world_and_simple_network()

        current_state = world.reset()
        bus = {'state': current_state}

        self.assertEqual(network.last_prediction_values, None)
        self.assertFalse(network.prediction_done)

        network.predict(bus, -1)
        self.assertTrue('action' in network.last_prediction_values)
        self.assertTrue('s1' in network.last_prediction_values)
        self.assertTrue(len(network.last_prediction_values) == 2)

    def test_add_experience(self):
        session, world, network = self.create_session_and_world_and_simple_network()
        current_state = world.reset()
        bus = {'state': current_state}
        network.predict(bus, -1)

        bus['next_state'] = world.step(0)[0]
        network.add_experience(bus, 123, False, 2)
        self.assertTrue('action' in network.last_prediction_values)
        self.assertTrue('s1' in network.last_prediction_values)
        self.assertTrue('s2' in network.last_prediction_values)
        self.assertTrue('reward' in network.last_prediction_values)
        self.assertEqual(network.last_prediction_values['reward'], 123)
        self.assertTrue('game_over' in network.last_prediction_values)
        self.assertEqual(network.last_prediction_values['game_over'], False)

        self.assertTrue(len(network.experiences) == 1)
        self.assertEqual(network.experiences[0]['reward'], 123)
        self.assertEqual(network.experiences[0]['game_over'], False)

        network.add_experience(bus, 456, False, 2)
        self.assertTrue(len(network.experiences) == 2)
        network.add_experience(bus, 789, True, 2)
        self.assertTrue(len(network.experiences) == 2)
        self.assertEqual(network.experiences[0]['reward'], 456)
        self.assertEqual(network.experiences[0]['game_over'], False)
        self.assertEqual(network.experiences[1]['reward'], 789)
        self.assertEqual(network.experiences[1]['game_over'], True)
