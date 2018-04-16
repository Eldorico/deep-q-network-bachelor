import tensorflow as tf

from world import *
from network import *
from state import *
from action import *
from agent import *



# create the world
world_config = {
    'ennemies' : True
}
world = World(world_config)
world.reset()

# create the session
session = tf.Session()

# use tensorboard
Global.USE_TENSORBOARD = True
Global.TENSORBOARD_DIR_NAME = '../TensorBoard/test01'
Global.SESSION = session

SAVE_FOLDER = '../Saves/avoid_ennemy_trainer'
PREFIX_FILES_NAME =  'test1'

# create the neural network that will learn to avoid ennemies
# avoid_ennemy_model = Model(session, 'avoid_ennemy', State.get_ennemy_agent_layer_shape(world), 1e-2,
#     [[64, 'relu'],
#     [32, 'relu'],
#     [Action.NB_POSSIBLE_ACTIONS, 'linear']]
# )
avoid_ennemy_model = ImportModel(session, SAVE_FOLDER, PREFIX_FILES_NAME, 'avoid_ennemy')
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

agent_config['save_folder'] = SAVE_FOLDER
agent_config['save_prefix_names'] = PREFIX_FILES_NAME

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
agent.train(world, 10000)
