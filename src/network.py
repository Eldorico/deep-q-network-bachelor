import numpy as np
import random
import tensorflow as tf
import os
import json
import sys

from action import *
from agent import *


ACTIVATIONS = {
    'tanh' : tf.nn.tanh,
    'relu' : tf.nn.relu,
    'elu'  : tf.nn.elu,
    'linear' : lambda x: x,
    'softmax' : tf.nn.softmax
}


class HiddenLayer:
    def __init__(self, model_name, size, layer_id, activation_function):
        self.id = layer_id
        self.W = tf.Variable(tf.random_normal(size, stddev=0.01), name=model_name+"_W_"+self.id)
        self.b = tf.Variable(tf.random_normal([size[1]], stddev=0.01), name=model_name+"_b_"+self.id)
        self.activation_function = activation_function
        self.size = size
        self.model_name = model_name

        if Global.USE_TENSORBOARD:
            self.W_histogram = tf.summary.histogram('hist_' + model_name+"_W_"+self.id, self.W)
            self.b_histogram = tf.summary.histogram('hist_' + model_name+"_b_"+self.id, self.b)

    def forward(self, X):
        with tf.name_scope(self.model_name +"_layer_"+self.id):
            Z = tf.matmul(X, self.W) + self.b
            return self.activation_function(Z)

class Model:
    def __init__(self, session, model_name, input_size, learning_rate, layers):
        self.learning_rate = learning_rate
        self.args = {'model_name': model_name, 'input_size': input_size, 'learning_rate': learning_rate, 'layers': layers} # keep the args in order to create a TargetModel easily
        self.output_size = layers[-1][0]
        self.name = model_name

        self.predict_op = None
        self.train_op = None
        self.session = session

        # set placeholders
        self.X = tf.placeholder(tf.float32, [None, input_size], name='X')
        self.T = tf.placeholder(tf.float32, [None, self.output_size], name='T')
        self.Y = tf.placeholder(tf.int32, [None, ], name='Y')

        # add layers
        self.layers = []
        for i, nb_neurons, activation in [ (i, n, a) for i, (n,a) in enumerate(layers)]:
            self.layers.append(HiddenLayer(self.name, [input_size, nb_neurons], str(i), ACTIVATIONS[activation]))
            input_size = nb_neurons

        # keep bias and weights to copy
        self.weights_and_biais = []
        for layer in self.layers:
            self.weights_and_biais += [layer.W, layer.b]

        # compute predict op
        with tf.name_scope(self.name+"_predict_op"):
            Z = self.X
            for layer in self.layers:
                Z = layer.forward(Z)
            self.predict_op = Z

        # compute train op
        with tf.name_scope(self.name+"_train_op"):
            Y = self.predict_op * tf.one_hot(self.Y, self.output_size)
            self.cost_op = tf.reduce_mean(tf.square(Y- self.T))
            # self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost_op)  # TODO: why using this optimizer it changes the weights?
            self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost_op)

            if Global.USE_TENSORBOARD:
                self.cost_scalar = tf.summary.scalar("cost", self.cost_op)

        # init the model variables (if we have a session. if we dont have a session, its because its an imported model so we dont want to init the variables. (session is set after in ImportModel()))
        if self.session is not None:
            variables_to_init = []
            for layer in self.layers:
                variables_to_init.append(layer.W)
                variables_to_init.append(layer.b)
            init = tf.variables_initializer(variables_to_init)
            self.session.run(init)

        # init tensorboard variables
        if Global.USE_TENSORBOARD:
            self.last_cost_summary_episode = -1

    def predict(self, input_values):
        predicted_values = self.session.run(self.predict_op, feed_dict={self.X: input_values})

        if Global.PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES > 0 and Global.EPISODE_NUMBER % Global.PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES == 0:
            max_index = np.argmax(predicted_values[0])
            max_value = predicted_values[0][max_index]
            min_index = np.argmin(predicted_values[0])
            min_value = predicted_values[0][min_index]
            print("predicted values: max: %d -> %f, min: %d -> %f" % (max_index, max_value, min_index, min_value))

        return predicted_values

    def train(self, X, Y, T):
        """
        :param: X : the input
        :param: Y (int): the action choosen. (size [None, 1])
        :param: T ([float]): the targets with ONLY the value to update. ex: [0,0,13.24, 0 0]
                the value to update corresponds to the choosen action. The size of T as to be [None, output_dim]
        """
        if Global.USE_TENSORBOARD and Global.EPISODE_NUMBER % Global.OUTPUT_TO_TENSORBOARD_EVERY_N_EPISODES == 0:
            _, cost_summary = self.session.run([self.train_op, self.cost_scalar], feed_dict={self.X:X, self.Y:Y, self.T:T})
            Global.WRITER.add_summary(cost_summary, Global.EPISODE_NUMBER)
            self.last_cost_summary_episode = Global.EPISODE_NUMBER
            # print("Predict = ")
            # print(train_result)
            # print("Cost = ")
            # print(cost_summary)
        else:
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

    def export_model(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save the config model file
        with open(folder + '/' + self.name + '.modelconfig', 'w') as outfile:
            self.args['name'] = self.name
            json.dump(self.args, outfile)
            self.args.pop('name')

        # save the graph and models
        saver = tf.train.Saver()
        saver.save(self.session, folder + '/' + self.name)

    def write_weights_tb_histograms(self):
        for layer in self.layers:
            W_summary = Global.SESSION.run(layer.W_histogram)
            Global.WRITER.add_summary(W_summary, Global.EPISODE_NUMBER)
            b_summary = Global.SESSION.run(layer.b_histogram)
            Global.WRITER.add_summary(b_summary, Global.EPISODE_NUMBER)

    def debug_list_all_variables(self):
        tvars = tf.trainable_variables()
        tvars_vals = self.session.run(tvars)

        for var, val in zip(tvars, tvars_vals):
            print(var.name, val)

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
        super().__init__(model.session, model.args['model_name']+'_tm', model.args['input_size'], model.args['learning_rate'], model.args['layers'])


class ImportModel(Model):
    def __init__(self, session, folder, model_name):
        base_path = folder + '/' + model_name

        # create the base model object
        try:
            with open( base_path + '.modelconfig', 'r') as config_file:
                args = json.load(config_file)
                super().__init__(None, args['model_name'], args['input_size'], args['learning_rate'], args['layers'])
        except (OSError, IOError) as e:
            sys.stderr.write("ImportModel(): import file not found: " + base_path + '.modelconfig\n')
            exit(-1)

        # restore the variables
        self.session = session
        saver = tf.train.Saver()
        saver.restore(self.session, base_path)


class Network:

    def __init__(self, model, input_adapter, continue_exploration=False, is_training=False):
        self.is_training = is_training
        self.model = model
        self.input_adapter = input_adapter
        self.explore = continue_exploration

        if self.is_training:
            self.target_model = TargetModel(self.model)

        self.prediction_done = False
        self.depends_on = []
        self.experiences = []
        self.last_prediction_values = None # {}

    def add_dependency(self, network):
        self.depends_on.append(network)

    def predict(self, bus, epsilon_value):
        choose_randomly = True if random.random() <= epsilon_value and self.explore else False

        # debug
        # if choose_randomly:
        #     print("episode %d: choosed randomly. epsilon_value = %f" % (Global.EPISODE_NUMBER, epsilon_value))

        if choose_randomly:
            action = random.randint(0,self.model.layers[-1].size[1]-1)
        else:
            input_value = self.input_adapter(bus)
            prediction = self.model.predict(input_value)[0]
            action = np.argmax(prediction)

        self.last_prediction_values = {'action' : action, 's1': self.input_adapter(bus) }
        self.prediction_done = True

        # debug
        # print("network.predict(): last_predictions_values_s1 = \n%s" % str(self.last_prediction_values['s1']))


    def add_experience(self, bus, reward, game_over, max_experience_size, add_only_n_last_steps=None):
        self.last_prediction_values['s2'] = self.input_adapter(bus, True)
        self.last_prediction_values['reward'] = reward
        self.last_prediction_values['game_over'] = game_over

        if add_only_n_last_steps is not None:
            if not hasattr(self, 'tmp_experiences'):
                self.tmp_experiences = []
            self.tmp_experiences.append(dict(self.last_prediction_values))
            if game_over:
                # print("Game over: added %d experiences to experience." % len(self.tmp_experiences[-add_only_n_last_steps:]))
                self.experiences += self.tmp_experiences[-add_only_n_last_steps:]
                self.tmp_experiences = []
        else:
            self.experiences.append(dict(self.last_prediction_values))

        while len(self.experiences) > max_experience_size:
            self.experiences.pop(0)

    def flush_last_prediction_var(self):
        self.last_prediction_values = None
        self.prediction_done = False

    def copy_target_network(self):
        self.target_model.copy_from(self.model)
        if Global.PRINT_TARGET_COPY_RATIO:
            print("Target network copied after having trained %d steps" % Global._NB_TRAINED_STEPS)
            Global._NB_TRAINED_STEPS = 0

    def train(self, gamma, min_experience_size, batch_size):
        if len(self.experiences) < min_experience_size:
            return

        batch =  [self.experiences.pop(random.randrange(len(self.experiences))) for _ in range(batch_size) ] # random.sample(self.experiences, batch_size)
        inputs = []
        choosen_actions = []
        targets = []
        for sample in batch:
            s1, s2, reward, game_over, action = sample['s1'], sample['s2'], sample['reward'], sample['game_over'], sample['action']

            inputs.append(s1[0])
            choosen_actions.append(action)

            action_values_s2 = self.target_model.predict(s2)[0]
            action_value = action_values_s2[action]
            target = self.target_model.predict(s1)[0]
            target[action] = reward if game_over else reward + gamma * action_value
            target = [0 if i != action else value for i, value in enumerate(target)]
            targets.append(target)

        self.model.train(inputs, choosen_actions, targets)
        if Global.SAY_WHEN_AGENT_TRAINED:
            print("Network trained")
        if Global.PRINT_TARGET_COPY_RATIO:
            Global._NB_TRAINED_STEPS += batch_size
