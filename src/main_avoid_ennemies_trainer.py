import tensorflow as tf

from world import *
from network import *
from state import *
from action import *
from agent import *

# create the world
def reward_function(world):
    if world.game_over:
        return - 5
    else:
        max_distance = 10
        security_distance = 5
        smallest_distance_ennemy_collision_course = float('Inf')
        for ennemy in world.ennemies:
            if Direction.is_in_collision_course(ennemy, world.agent, security_distance):
                distance = Direction.distance(ennemy, world.agent)
                if distance < smallest_distance_ennemy_collision_course:
                    smallest_distance_ennemy_collision_course = distance
        if smallest_distance_ennemy_collision_course >= max_distance:
            return 1
        elif smallest_distance_ennemy_collision_course <=2:
            return 0.01 * smallest_distance_ennemy_collision_course
        else:
            return ((smallest_distance_ennemy_collision_course -2) /max_distance) ** 0.4
world_config = {
    'ennemies' : True,
    'print_reward' : False,
    'reward_function': reward_function
}
world = World(world_config)
world.reset()

# create the session
session = tf.Session()

# use tensorboard
Global.USE_TENSORBOARD = True
Global.SAVE_FOLDER = '../tmp_saves/avoid_ennemy_toy_trainer/reward_test_5'
Global.SESSION = session

# debug
Global.PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES = 10000
Global.PRINT_REWARD_EVERY_N_EPISODES = 10000
Global.PRINT_EPISODE_NB_EVERY_N_EPISODES = 2500
Global.PRINT_SCORE_AVG_EVERY_N_EPISODES = 5000
Global.SAY_WHEN_HISTOGRAMS_ARE_PRINTED = False
Global.SAY_WHEN_AGENT_TRAINED = False

# create the neural network that will learn to avoid ennemies
avoid_ennemy_model = Model(session, 'avoid_ennemy', State.get_ennemy_agent_layer_shape(world), 1e-2,
        [[64, 'relu'],
        [Action.NB_POSSIBLE_ACTIONS, 'linear']]
)
# avoid_ennemy_model = ImportModel(session, Global.SAVE_FOLDER, 'avoid_ennemy')
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
epsilon = Epsilon(0.1)
def update_epsilon(epsilon):
    epsilon.value = epsilon.value
epsilon.set_epsilon_function(update_epsilon)

agent_config = {}
agent_config['epsilon'] = epsilon
agent_config['networks'] = [avoid_ennemy_network]
agent_config['output_network'] = avoid_ennemy_network
agent_config['copy_target_period'] = 1000
agent_config['min_experience_size'] = 1000
agent_config['max_experience_size'] = 5000
agent_config['batch_size'] = 256
agent_config['gamma'] = 0.9

agent = Agent(agent_config)

# create the sig Int handler
import signal
import sys
def signal_handler(signal, frame):
        print('You pressed Ctrl+C!')
        print('Sending signal to agent...')
        agent.send_exit_signal()
signal.signal(signal.SIGINT, signal_handler)

# train agent for avoiding ennemies
agent.train(world, 1000000)
