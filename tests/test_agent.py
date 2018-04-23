import os
import sys
sys.path.append('../src')
import numpy as np
import unittest

from network import *
from world import *
from state import *
from agent import *



class AgentTest(unittest.TestCase):

    def create_simple_agent(self):
        # create the world
        world_config = {
            'ennemies' : True
        }
        world = World(world_config)
        world.reset()

        # create the session
        session = tf.Session()

        # create the neural network that will learn to avoid ennemies
        avoid_ennemy_model = Model(session, 'avoid_ennemy_network', State.get_ennemy_agent_layer_shape(world), 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [Action.NB_POSSIBLE_ACTIONS, 'linear']]
        )
        def avoid_ennemy_input_adapter(bus, next_state=False):
            if next_state:
                return bus['next_state'].get_ennemy_agent_layer_only()
            else:
                return bus['state'].get_ennemy_agent_layer_only()
        avoid_ennemy_network = Network(
            avoid_ennemy_model,
            avoid_ennemy_input_adapter,
            True,
            True
        )

        # create agent and his hyperparameters (config)
        epsilon = Epsilon(0.05)
        def update_epsilon(epsilon):
            epsilon.value = epsilon.value
        epsilon.set_epsilon_function(update_epsilon)

        agent_config = {}
        agent_config['epsilon'] = epsilon
        agent_config['networks'] = [avoid_ennemy_network]
        agent_config['output_network'] = avoid_ennemy_network
        agent_config['copy_target_period'] = 100
        agent_config['min_experience_size'] = 1000
        agent_config['max_experience_size'] = 5000
        agent_config['batch_size'] = 256
        agent_config['gamma'] = 0.9

        agent = Agent(agent_config)
        return world, agent

    def test_states_experiences_of_agent_are_different_on_one_episode(self):
        world, agent = self.create_simple_agent()
        agent.train(world, 1)

        # test that some states recorded are different
        self.assertFalse(np.array_equal(agent.networks[0].experiences[0]['s1'],agent.networks[0].experiences[1]['s1']))
        self.assertFalse(np.array_equal(agent.networks[0].experiences[0]['s2'],agent.networks[0].experiences[1]['s2']))
        self.assertFalse(np.array_equal(agent.networks[0].experiences[1]['s1'],agent.networks[0].experiences[2]['s1']))
        self.assertFalse(np.array_equal(agent.networks[0].experiences[1]['s2'],agent.networks[0].experiences[2]['s2']))
        self.assertFalse(np.array_equal(agent.networks[0].experiences[0]['s1'],agent.networks[0].experiences[2]['s1']))
        self.assertFalse(np.array_equal(agent.networks[0].experiences[0]['s2'],agent.networks[0].experiences[2]['s2']))

    #def test_training_is_good(self):
        
