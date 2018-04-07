
from keras.models import Sequential
from keras.layers import Dense, Activation
from world import *
from network import *
from state import *
from action import *

bus = {}

# create the world
world_config = {
    'ennemies' : True
}
world = World(world_config)
state = world.reset()

# create the neural network that will learn to avoid ennemies
avoid_ennemy_model = Sequential([
    Dense(64, input_dim=State.get_ennemy_agent_layer_shape(world)),
    Activation('relu'),
    Dense(32),
    Activation('relu'),
    Dense(Action.NB_POSSIBLE_ACTIONS),
    Activation('linear'),
])
avoid_ennemy_model.compile(optimizer='adam',
              loss='mean_squared_error')
def avoid_ennemy_input_adapter(bus):
    return bus['state'].get_ennemy_agent_layer_only()
avoid_ennemy_network = Network(
    avoid_ennemy_model,
    avoid_ennemy_input_adapter,
    True,
    True
)

# agent hyperparameters (config)
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
