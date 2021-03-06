import tensorflow as tf

from world import *
from network import *
from state import *
from action import *
from agent import *

# create the world
def reward_function(world):
    # if world.game_over:
    #     return -1
    # elif world.food.found:
    #     return 1
    # else:
    #     return -0.01
    if world.game_over:
        return - 5
    max_distance = 73
    return (1 - Direction.distance(world.agent, world.food) / max_distance)
world_config = {
    'food' : True,
    'print_reward' : False,
    'reward_function': reward_function
}
world = World(world_config)
world.reset()

# create the session
session = tf.Session()

# use tensorboard
Global.USE_TENSORBOARD = True
Global.SAVE_MAIN_FILE = True
Global.SAVE_FOLDER = '../tmp_saves/food3/good_episodes_kind_reward_gamma05_first_good_steps_alpha_e-1'
Global.SESSION = session

# debug
Global.PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES = 5000 # 10000
# Global.PRINT_REWARD_EVERY_N_EPISODES = 10000
Global.PRINT_EPISODE_NB_EVERY_N_EPISODES = 2500
Global.PRINT_SCORE_AVG_EVERY_N_EPISODES = 500
Global.SAY_WHEN_HISTOGRAMS_ARE_PRINTED = False
Global.SAY_WHEN_AGENT_TRAINED = False
Global.OUTPUT_TO_TENSORBOARD_EVERY_N_EPISODES = 500

# create the neural network that will learn to fetch
fetch_object_model = Model(session, 'fetch_object', 4, 1e-1,
        [[40, 'relu'],
         [40, 'relu'],
        [Action.NB_POSSIBLE_MOVE_ACTION, 'linear']]
)
# fetch_object_model = ImportModel(session, Global.SAVE_FOLDER, 'fetch_object')
def fetch_object_input_adapter(bus, next_state=False):
    index = 'next_state' if next_state else 'state'
    agent_position = bus[index].get_agent_position_layer()
    food_position = bus[index].get_food_position_layer()
    return [np.array([agent_position, food_position]).flatten()]

def fetch_object_add_experience_hook(network, world):
    if not hasattr(network, 'tmp_experiences'):
        network.tmp_experiences = []
    network.tmp_experiences.append(dict(network.last_prediction_values))
    if world.game_over and world.score > 110:
        network.experiences += network.tmp_experiences[:-99]
        network.tmp_experiences = []

fetch_object_network = Network(
    fetch_object_model,
    fetch_object_input_adapter,
    True,
    True,
    fetch_object_add_experience_hook
)

# create agent and his hyperparameters (config)
# epsilon = Epsilon(1)
epsilon = Epsilon(0.1)
def update_epsilon(epsilon):
    # epsilon.value = max(0.1, epsilon.value - 1e-6)
    epsilon.value = epsilon.value
epsilon.set_epsilon_function(update_epsilon)

agent_config = {}
agent_config['epsilon'] = epsilon
agent_config['networks'] = [fetch_object_network]
agent_config['output_network'] = fetch_object_network
agent_config['copy_target_period'] = 10000
agent_config['min_experience_size'] = 50000
agent_config['max_experience_size'] = 400000
agent_config['batch_size'] = 32
agent_config['gamma'] = 0.5

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
agent.train(world, 5000000)
