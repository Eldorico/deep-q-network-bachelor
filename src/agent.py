import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np

class Global:
    USE_TENSORBOARD = False
    SAVE_FOLDER = None
    TB_FOLDER = None
    SESSION = None
    WRITER = None
    EPISODE_NUMBER = 0

class Epsilon:
    def __init__(self, start_epsilon_value):
        self.value = start_epsilon_value

    def set_epsilon_function(self, epsilon_function):
        self.epsilon_function = epsilon_function

    def update_epsilon(self):
        self.epsilon_function(self)

class Agent:

    def __init__(self, config, session=None):
        self.networks = config['networks']
        self.output_network = config['output_network']
        self.copy_target_period = config['copy_target_period']
        self.min_experience_size = config['min_experience_size']
        self.max_experience_size = config['max_experience_size']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']

        self.exit = False

        if Global.USE_TENSORBOARD:
            Global.EPISODE_NUMBER = 0

            self.writer = tf.summary.FileWriter(Global.SAVE_FOLDER)
            self.writer.add_graph(Global.SESSION.graph)
            Global.WRITER = self.writer

            self.actions_made_placeholder = tf.placeholder(tf.int32, [None, 1])
            self.actions_made_histogram = tf.summary.histogram('actions_distribution', self.actions_made_placeholder)

        self.bus = {} # used to keep the current_state and the next_state (s1 and s2)

        self.nb_steps_played = 0

    def send_exit_signal(self):
        self.exit = True

    def _save(self):
        if Global.SAVE_FOLDER is not None:
            print("Saving networks models as %s ..." % (Global.SAVE_FOLDER+'/'))
            for network in self.networks:
                network.model.export_model(Global.SAVE_FOLDER)
            print("Saving done.")

    def _save_and_exit(self):
        self._save()
        print('Exiting now')
        exit(0)

    def play_episode(self, world, episode_number=None):
        current_state = world.reset()
        action_log = []

        while not world.game_over:
            action = self.choose_action(current_state)
            next_state, reward, game_over, world_informations = world.step(action)

            if self.nb_steps_played % self.copy_target_period == 0:
                self.copy_target_networks()

            self.add_experience(next_state, reward, game_over)
            self.flush_last_prediction_var()
            self.train_networks()

            current_state = next_state
            self.epsilon.update_epsilon()
            self.nb_steps_played += 1
            action_log.append(action)

        Global.EPISODE_NUMBER += 1
        return {'score': world_informations['score'], 'actions_made' : action_log}

    def flush_last_prediction_var(self):
        """ Removes the last prediction_values of the networks and sets their
            prediction_done variable to False, so the networks can redo a new prediction
            after that.
        """
        for network in self.networks:
            network.flush_last_prediction_var()

    def choose_action(self, state):
        self.bus['state'] = state

        nb_networks_that_predicted = 0
        while nb_networks_that_predicted != len(self.networks):
            for network in self.networks:
                if len(network.depends_on) is 0 or all(dependency.prediction_done for dependency in network.depends_on):
                    network.predict(self.bus, self.epsilon.value)
                    nb_networks_that_predicted += 1

        return self.output_network.last_prediction_values['action']

    def add_experience(self, next_state, reward, game_over):
        """ add experiences to the networks that are training
        """
        self.bus['next_state'] = next_state
        for network in self.networks:
            if network.is_training:
                network.add_experience(self.bus, reward, game_over, self.max_experience_size)

    def train_networks(self):
        for network in self.networks:
            if network.is_training:
                network.train(self.gamma, self.min_experience_size, self.batch_size)

    def copy_target_networks(self):
        """ Make the network that are training do a copy of themselves.
            (refresh TNetworks if the Network is in training mode)
        """
        for network in self.networks:
            if network.is_training:
                network.copy_target_network()

    def train(self, world, nb_episodes, avg_every_n_episodes=100, stop_on_score_avg=None):
        score_avg = 0
        tmp_total_score = 0

        if Global.USE_TENSORBOARD:
            log = {'actions_made' : []}

        for i in range(nb_episodes):
            print("episode %d" % (i+1))

            results = self.play_episode(world, i)
            tmp_total_score += results['score']

            # update tensorboard
            if Global.USE_TENSORBOARD:
                value = summary_pb2.Summary.Value(tag="score_per_episode", simple_value=results['score'])
                summary = summary_pb2.Summary(value=[value])
                self.writer.add_summary(summary, i)

                value = summary_pb2.Summary.Value(tag="epsilon_value", simple_value=self.epsilon.value)
                summary = summary_pb2.Summary(value=[value])
                self.writer.add_summary(summary, i)

                log['actions_made'] += results['actions_made']
                if i % 50 == 0 and i != 0:
                    summary = Global.SESSION.run(self.actions_made_histogram, feed_dict={self.actions_made_placeholder: np.reshape(log['actions_made'], (len(log['actions_made']), 1))})
                    self.writer.add_summary(summary, i)
                    log['actions_made'] = []

            # check if avg score is reached
            if i % avg_every_n_episodes == 0 and i != 0:
                score_avg = tmp_total_score / avg_every_n_episodes
                print("score avg after %d episodes: %f" % (i, score_avg) )
                tmp_total_score = 0
                if stop_on_score_avg is not None and score_avg >= stop_on_score_avg:
                    print("Score avg reached. Stop learning")
                    self.exit = True

            # exit
            if self.exit:
                self._save_and_exit()

        print("Nb max episodes reached. Stop learning")
        if Global.SAVE_FOLDER is not None:
            self._save()
