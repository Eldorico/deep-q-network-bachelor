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

# DANGER ZONE: MODIFY ONLY THIS IF YOU ARE CREATING THE CURRENT AGENT!!!!! ELSE: leave it like this in order to import the same agent
SAVE_FOLDER = '../tmp_saves/avoid_ennemy_trainer/my_test_folder'
Global.FIRST_TIME_CREATING_AGENT = True

# make sure the agent is created for the first time
# TODO: ...

# create the session
session = tf.Session()

# use tensorboard
Global.USE_TENSORBOARD = True
Global.SAVE_FOLDER = SAVE_FOLDER
Global.SESSION = session

# create the neural network that will learn to avoid ennemies
AVOID_ENENEMY_NETWORK_NAME = 'avoid_ennemy'
if FIRST_TIME_CREATING_AGENT:
    avoid_ennemy_model = Model(session, AVOID_ENENEMY_NETWORK_NAME, State.get_ennemy_agent_layer_shape(world), 1e-2,
        [[64, 'relu'],
        [32, 'relu'],
        [Action.NB_POSSIBLE_ACTIONS, 'linear']]
    )
else:
    avoid_ennemy_model = ImportModel(session, SAVE_FOLDER, '', AVOID_ENENEMY_NETWORK_NAME)
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

# train agent for avoiding ennemies
agent.train(world, 15000)
