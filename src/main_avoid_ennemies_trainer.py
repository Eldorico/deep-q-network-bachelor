
from keras.models import Sequential
from keras.layers import Dense, Activation
from world import *
from network import *
from state import *
from action import *


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
def avoid_ennemy_input_adapter(state):
    return state.get_ennemy_agent_layer_only()
avoid_ennemy_network = Network(
    avoid_ennemy_model,
    avoid_ennemy_input_adapter,
    True,
    True
)


# debug
print(State.get_ennemy_agent_layer_shape(world))
prediction = avoid_ennemy_network.predict(state)
print(prediction)
print("Weights: ")
print(avoid_ennemy_network.model.layers[-2].get_weights())
print("Weights target: ")
print(avoid_ennemy_network.target_model.layers[-2].get_weights())
print("ID checks")
print(id(avoid_ennemy_network.model))
print(id(avoid_ennemy_network.target_model))
