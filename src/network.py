import numpy as np
import random
from action import *
import tensorflow as tf
import os
import json


ACTIVATIONS = {
    'tanh' : tf.nn.tanh,
    'relu' : tf.nn.relu,
    'elu'  : tf.nn.elu,
    'linear' : lambda x: x
}


class HiddenLayer:
    def __init__(self, size, layer_id, activation_function):
        self.id = layer_id
        self.W = tf.Variable(tf.random_normal(size, stddev=0.01), name="W_"+self.id)
        self.b = tf.Variable(tf.random_normal([size[1]], stddev=0.01), name="b_"+self.id)
        self.activation_function = activation_function
        self.size = size

    def forward(self, X):
        with tf.name_scope("layer_"+self.id):
            Z = tf.matmul(X, self.W) + self.b
            return self.activation_function(Z)

class Model:
    def __init__(self, input_size, learning_rate, layers):
        self.learning_rate = learning_rate
        self.args = {'input_size': input_size, 'learning_rate': learning_rate, 'layers': layers} # keep the args in order to create a TargetModel easily
        self.output_size = layers[-1][0]
        self.name = None

        self.predict_op = None
        self.train_op = None
        self.session = None


        # set placeholders
        self.X = tf.placeholder(tf.float32, [None, input_size], name='X')
        self.T = tf.placeholder(tf.float32, [None, self.output_size], name='T')
        self.Y = tf.placeholder(tf.int32, [None, ], name='Y')

        # add layers
        self.layers = []
        for i, nb_neurons, activation in [ (i, n, a) for i, (n,a) in enumerate(layers)]:
            self.layers.append(HiddenLayer([input_size, nb_neurons], str(i), ACTIVATIONS[activation]))
            input_size = nb_neurons


        # keep bias and weights to copy
        self.weights_and_biais = []
        for layer in self.layers:
            self.weights_and_biais += [layer.W, layer.b]

        # compute predict op
        with tf.name_scope("predict_op"):
            Z = self.X
            for layer in self.layers:
                Z = layer.forward(Z)
            self.predict_op = Z

        # compute train op
        with tf.name_scope("train_op"):
            Y = self.predict_op * tf.one_hot(self.Y, self.output_size)
            self.cost_op = tf.reduce_mean(tf.square(Y- self.T))
            # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost_op)  # TODO: why using this optimizer it changes the weights?
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_op)

    def set_name(self, name):
        self.name = name

    def predict(self, input_values):
        return self.session.run(self.predict_op, feed_dict={self.X: input_values})

    def train(self, X, Y, T):
        """
        :param: X : the input
        :param: Y (int): the action choosen. (size [None, 1])
        :param: T ([float]): the targets with ONLY the value to update. ex: [0,0,13.24, 0 0]
                the value to update corresponds to the choosen action. The size of T as to be [None, output_dim]
        """
        self.session.run(
            self.train_op,
            feed_dict = {
                self.X : X,
                self.Y : Y,
                self.T : T
            }
        )

    def copy_from(self, other):
        updates_to_run = []
        my_params = self.weights_and_biais
        other_params = other.weights_and_biais
        for my_param, other_param in zip(my_params, other_params):
            other_values = self.session.run(other_param)
            value_to_update = my_param.assign(other_values)
            updates_to_run.append(value_to_update)
        self.session.run(updates_to_run)

    def export_model(self, folder, filesname):
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save the config model file
        with open(folder + '/' + filesname + '.modelconfig', 'w') as outfile:
            self.args['name'] = self.name
            json.dump(self.args, outfile)

        # save the graph and models
        saver = tf.train.Saver()
        saver.save(self.session, folder + '/' + filesname)

    def set_session(self, session):
        self.session = session

    def debug_return_cost(self, X, Y, T):
        # print("Model.debug_return_cost(): Y=%s, T=%s")
        return self.session.run(
            self.cost_op,
            feed_dict = {
                self.X : X,
                self.Y : Y,
                self.T : T
            }
        )

class TargetModel(Model):
    def __init__(self, model):
        super().__init__(model.args['input_size'], model.args['learning_rate'], model.args['layers'])


class Network:

    def __init__(self, name, model, input_adapter=None, continue_exploration=False, is_training=False):
        self.name = name
        self.is_training = is_training
        self.model = model
        self.model.set_name(name)
        self.input_adapter = input_adapter
        self.explore = continue_exploration

        if self.is_training:
            self.target_model = keras.models.clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

        self.prediction_done = False
        self.depends_on = []
        self.experiences = []
        self.last_prediction_values = None # {}

    def add_dependency(self, network):
        self.depends_on.append(network)

    def predict(self, bus, epsilon):
        choose_randomly = True if random.random() <= epsilon and self.explore else False

        if choose_randomly:
            action = random.randint(0, Action.NB_POSSIBLE_ACTIONS -1)
        else:
            input_value = self.input_adapter(bus)
            prediction = self.model.predict(input_value)[0]
            action = np.argmax(prediction)

        self.last_prediction_values = {'action' : action, 's1': self.input_adapter(bus) }
        self.prediction_done = True

        # debug
        # return prediction

    def add_experience(self, bus, reward, game_over, max_experience_size):
        self.last_prediction_values['s2'] = self.input_adapter(bus, True)
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

    def train(self, gamma, min_experience_size, batch_size, tensorboard=None):
        if len(self.experiences) < min_experience_size:
            return

        batch = random.sample(self.experiences, batch_size)
        inputs = []
        targets = []
        for sample in batch:
            s1, s2, reward, game_over, action = sample['s1'], sample['s2'], sample['reward'], sample['game_over'], sample['action']

            inputs.append(s1[0])

            action_values_s2 = self.target_model.predict(s2)[0]
            action_value = action_values_s2[action]
            target = self.target_model.predict(s1)
            target[0][action] = reward if game_over else reward + gamma * action_value
            target = [0 if i != action else value for i, value in enumerate(target[0])]
            targets.append(target[0])

        self.model.fit(np.array(inputs), np.array(targets), batch_size=len(inputs), epochs=1, verbose=0)
