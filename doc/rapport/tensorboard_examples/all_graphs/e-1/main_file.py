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
        safe_distance = 20
        min_distance = float('inf')
        for ennemy in world.ennemies:
            distance = Direction.distance(ennemy, world.agent)
            if distance < min_distance:
                min_distance = distance

        if min_distance >= safe_distance:
            return math.log(safe_distance+0.01) -1
        elif min_distance <= 1:
            return -1
        else:
            return math.log(min_distance+0.01) -1
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
Debug.USE_TENSORBOARD = True
Debug.SAVE_MAIN_FILE = True
Debug.SAVE_FOLDER = '../tmp_saves/last_debug/avoid_ennemies'
Debug.SESSION = session

# debug
# Debug.PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES = 10000
Debug.PRINT_EPISODE_NB_EVERY_N_EPISODES = 2500
Debug.PRINT_SCORE_AVG_EVERY_N_EPISODES = 50 # 5000
Debug.SAY_WHEN_HISTOGRAMS_ARE_PRINTED = False
Debug.SAY_WHEN_AGENT_TRAINED = False
Debug.OUTPUT_TO_TENSORBOARD_EVERY_N_EPISODES = 5000

# create the neural network that will learn to avoid ennemies
avoid_ennemy_model = Model(session, 'avoid_ennemy', State.get_ennemy_agent_layer_shape(world)*3, 1e-1,
       [[40, 'relu'],
        [40, 'relu'],
       [Action.NB_POSSIBLE_MOVE_ACTION, 'linear']]
)
# avoid_ennemy_model = ImportModel(session, Debug.SAVE_FOLDER, 'avoid_ennemy')
def avoid_ennemy_input_adapter(bus, next_state=False):
    if next_state:
        input_states = [state.get_ennemy_agent_layer_only() for state in bus['last_states'][1:]]
    else:
        input_states = [state.get_ennemy_agent_layer_only() for state in bus['last_states'][0:3]]
    input_states = [np.array(input_states).flatten()]
    return input_states
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
agent_config['copy_target_period'] = 10000
agent_config['min_experience_size'] = 50000
agent_config['max_experience_size'] = 400000
agent_config['batch_size'] = 32
agent_config['gamma'] = 0.8

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
