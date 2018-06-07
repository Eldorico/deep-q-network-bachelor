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
    'food' : True,
    'print_reward' : False,
    'reward_function': reward_function,
    'render' : True
}
world = World(world_config)

# create the session
session = tf.Session()

# use tensorboard
Debug.SAVE_FOLDER = '../saves/fetch_object_backup'
Debug.SESSION = session

print("loading save from folder %s" % Debug.SAVE_FOLDER)

# create the neural network that will learn to avoid ennemies
avoid_ennemy_model = ImportModel(session, Debug.SAVE_FOLDER, 'fetch_object')
def fetch_object_input_adapter(bus, next_state=False):
    index = 'next_state' if next_state else 'state'
    agent_position = bus[index].get_agent_position_layer()
    food_position = bus[index].get_food_position_layer()
    return [np.array([agent_position, food_position]).flatten()]
avoid_ennemy_network = Network(
    avoid_ennemy_model,
    fetch_object_input_adapter,
)

# create agent and his hyperparameters (config)
epsilon = Epsilon(0)
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

# see the agent result
for i in range(10):
    print("episode %d" % i)
    results = agent.play_episode(world, 500)
    print("score = %d" % results['score'])
