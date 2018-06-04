import tensorflow as tf

from world import *
from network import *
from state import *
from action import *
from agent import *

# create the world
def reward_function(world):
    if world.game_over:
        return - 1
    else:
        return 0.002  # goal is 500 score. So i'lltry a reward of 1/500 for each step
world_config = {
    'ennemies' : True,
    'food': True,
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
Global.SAVE_FOLDER = '../tmp_saves/play_game/asdf'
Global.SESSION = session

# debug
# Global.PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES = 10000
# Global.PRINT_REWARD_EVERY_N_EPISODES = 10000
Global.PRINT_EPISODE_NB_EVERY_N_EPISODES = 2500
Global.PRINT_SCORE_AVG_EVERY_N_EPISODES = 5000
Global.SAY_WHEN_HISTOGRAMS_ARE_PRINTED = False
Global.SAY_WHEN_AGENT_TRAINED = False
Global.OUTPUT_TO_TENSORBOARD_EVERY_N_EPISODES = 5000

# load the neural network that know how avoid ennemies
avoid_ennemy_model = ImportModel(session, Global.SAVE_FOLDER, 'avoid_ennemy')
def avoid_ennemy_input_adapter(bus, next_state=False):
    if next_state:
        input_states = [state.get_ennemy_agent_layer_only() for state in bus['last_states'][1:]]
    else:
        input_states = [state.get_ennemy_agent_layer_only() for state in bus['last_states'][0:3]]
    input_states = [np.array(input_states).flatten()]
    return input_states
avoid_ennemy_network = Network(
    avoid_ennemy_model,
    avoid_ennemy_input_adapter
)

# load the neural that know how to fetch food
fetch_food_model = ImportModel(session, Global.SAVE_FOLDER, 'fetch_object')
def fetch_food_input_adapter(bus, next_state=False):
    index = 'next_state' if next_state else 'state'
    agent_position = bus[index].get_agent_position_layer()
    food_position = bus[index].get_food_position_layer()
    return [np.array([agent_position, food_position]).flatten()]
fetch_food_network = Network(
    fetch_food_model,
    fetch_food_input_adapter
)

# create the network model that will learn to play the game using the fetch food and avoid ennemies networks
agent_ennemies_input_size = State.get_ennemy_agent_layer_shape(world)*3
food_position_size = 2
stamina_size = 1
play_game_input_size = agent_ennemies_input_size + food_position_size + stamina_size
play_game_model = Model(session, 'play_game', play_game_input_size, 1e-1,
        [[40, 'relu'],
         [40, 'relu'],
        [1, 'sigmoid']]  # output = 1 : if output < 0.5 => play avoid ennemy move. Else play fetch food move.
)
def play_game_input_adapter(bus, next_state=False):
    pass
def play_game_output_adapter():
    pass

play_game_network = Network(
    play_game_model,
    play_game_input_adapter,
    True,
    True,
    None,
    play_game_output_adapter
)
play_game_network.add_dependency(fetch_food_network)
play_game_network.add_dependency(avoid_ennemy_network)

# create agent and his hyperparameters (config)
epsilon = Epsilon(0.1)
def update_epsilon(epsilon):
    epsilon.value = epsilon.value
epsilon.set_epsilon_function(update_epsilon)

agent_config = {}
agent_config['epsilon'] = epsilon
agent_config['networks'] = [avoid_ennemy_network, fetch_food_network, play_game_network]
agent_config['output_network'] = play_game_network
agent_config['copy_target_period'] = 10000
agent_config['min_experience_size'] = 50000
agent_config['max_experience_size'] = 400000
agent_config['batch_size'] = 32
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
agent.train(world, 5000000)
