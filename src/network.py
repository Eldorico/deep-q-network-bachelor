import numpy as np
import keras
import random

class Network:

    def __init__(self, model, input_adapter=None, is_training=False):
        self.is_training = is_training
        self.model = model
        self.input_adapter = input_adapter

        if self.is_training:
            self.target_model = keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

        self.prediction_done = False
        self.depends_on = []
        self.experiences = []
        self.last_prediction_values = None # {}

    def add_dependency(network):
        self.depends_on.append(network)

    def predict(self, state):
        input_value = self.input_adapter(state)
        prediction = self.model.predict(input_value)
        action = np.argmax(prediction)
        self.last_prediction_values = {'action' : action, 's1': self.input_adapter(state) }
        self.prediction_done = True

        # debug
        return prediction

    def add_experience(self, next_state, reward, game_over, max_experience_size):
        self.last_prediction_values['s2'] = self.input_adapter(next_state)
        self.last_prediction_values['reward'] = reward
        self.last_prediction_values['game_over'] = game_over

        self.experiences.append(self.last_prediction_values)

        if len(self.experiences) > max_experience_size:
            self.experiences.pop(0)

    def flush_last_prediction_var(self):
        self.last_prediction_values = None
        self.prediction_done = False

    def copy_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def train(self, gamma, min_experience_size, batch_size):
        if len(self.experiences) < min_experience_size:
            return

        batch = random.sample(self.experiences, batch_size)
        targets = []
        inputs = []
        for sample in batch:
            s1, s2, reward, game_over, action = sample['s1'], sample['s2'], sample['reward'], sample['game_over'], sample['action']

            inputs.append(s1)

            action_values_s2 = self.target_model.predict(s2)
            action_value = action_values_s2[action]
            target = self.target_model.predict(s1)
            target[action] = reward if game_over else reward + gamma * action_value
            targets.append(target)

        self.model.fit(inputs, targets, epochs=1, verbose=0)
