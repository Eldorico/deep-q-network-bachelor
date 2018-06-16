import tensorflow as tf

from world import *
from network import *
from state import *
from action import *
from agent import *
from debugger import *

# create the world
def reward_function(world):
    if world.game_over:
        return - 5
    max_distance = 73
    return (1 - Direction.distance(world.agent, world.food) / max_distance)
world_config = {
    'ennemies' : True,
    'food': True,
    'print_reward' : False,
    'reward_function': reward_function
}
world = World(world_config)
world.reset()

# create the training session
session = tf.Session()

# use tensorboard
Debug.USE_TENSORBOARD = True
Debug.SAVE_MAIN_FILE = True
Debug.SAVE_FOLDER = '../tmp_saves/play_game'
Debug.SESSION = session

# config the debugger
Debug.PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES = 1000
Debug.PRINT_PREDICTED_VALUES_FOR.append('play_game')
Debug.PRINT_EPISODE_NB_EVERY_N_EPISODES = 500 # 2500
Debug.PRINT_SCORE_AVG_EVERY_N_EPISODES = 500
Debug.SAY_WHEN_HISTOGRAMS_ARE_PRINTED = False
Debug.SAY_WHEN_AGENT_TRAINED = False
Debug.OUTPUT_TO_TENSORBOARD_EVERY_N_EPISODES = 500

# load the neural network that knows how avoid ennemies
avoid_ennemy_model = ImportModel(None, Debug.SAVE_FOLDER, 'avoid_ennemy')
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

# load the neural that knows how to fetch food
fetch_food_model = ImportModel(None, Debug.SAVE_FOLDER, 'fetch_object')
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
play_game_model = Model(session, 'play_game', 3, 1e-2,
        [[40, 'relu'],
         [40, 'relu'],
        [2, 'linear']]
)
# play_game_model = ImportModel(session, Debug.SAVE_FOLDER, 'play_game')
def play_game_input_adapter(bus, next_state=False):
    index = 'next_state' if next_state else 'state'
    stamina = bus[index].get_stamina_value()
    distance_from_ennemy = bus[index].get_min_distance_between_agent_ennemy()
    distance_from_food = bus[index].get_distance_from_food()
    return np.array([[stamina, distance_from_ennemy, distance_from_food]])
def play_game_output_adapter(action):
    if action == 0:
        return avoid_ennemy_network.last_prediction_values['action']
    else:
        return fetch_food_network.last_prediction_values['action']

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
agent.train(world, 1000000)
