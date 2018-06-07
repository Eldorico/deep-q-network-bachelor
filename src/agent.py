import numpy as np

from state import *
from action import Action
from debugger import *


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

        Debug.set_episode_number(0)
        Debug.create_tensorboard_writer()
        self.actions_made_placeholder, self.actions_made_histogram = Debug.create_tensorboard_writer() # TODO: can remove this attributes since they are on Debug?
        # save the main file before someone change it by error
        Debug.save_main_file()

        self.bus = {} # used to keep the current_state and the next_state (s1 and s2)
        self.nb_steps_played = 0

    def send_exit_signal(self):
        self.exit = True

    def _save(self):
        if Debug.SAVE_FOLDER is not None:
            for network in filter(lambda x: x.is_training, self.networks):
                print("Saving network %s model as %s ..." % (network.model.name, Debug.SAVE_FOLDER + '/') )
                network.model.export_model(Debug.SAVE_FOLDER)
            print("Saving done.")

    def _save_and_exit(self):
        self._save()
        print('Exiting now')
        exit(0)

    def play_episode(self, world, max_episode_steps=None):
        current_state = world.reset()
        action_log = []
        episode_nb_steps = 0
        self.bus['last_states'] = [current_state for i in range(4)] # if we want to use the 3/4 lasts steps in input adapter. since at first we dont have the last steps, we copy the first step 3/4 times

        episode_done = False
        while not episode_done:
            action = self.choose_action(current_state)
            next_state, reward, game_over, world_debug = world.step(action)

            if self.nb_steps_played % self.copy_target_period == 0:
                self.copy_target_networks()
                Debug.print_target_network_copied()

            self.add_experience(next_state, reward, game_over, world)
            self.flush_last_prediction_var()
            self.train_networks()

            if game_over or (max_episode_steps is not None and max_episode_steps <= episode_nb_steps):
                episode_done = True

            current_state = next_state
            self.epsilon.update_epsilon()
            self.nb_steps_played += 1
            episode_nb_steps += 1
            action_log.append(action)

        Debug.increment_episode_number()
        return {'score': world_debug['score'], 'total_reward': world_debug['total_reward'] , 'actions_made' : action_log}

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

        return self.output_network.get_last_action()

    def add_experience(self, next_state, reward, game_over, world):
        """ add experiences to the networks that are training
        """
        self.bus['next_state'] = next_state
        self.bus['last_states'].pop(0)
        self.bus['last_states'].append(next_state)

        for network in self.networks:
            if network.is_training:
                network.add_experience(self.bus, reward, game_over, self.max_experience_size, world)

    def train_networks(self):
        Debug.pause_non_training_chrono_and_resume_training_chrono()

        for network in self.networks:
            if network.is_training:
                network.train(self.gamma, self.min_experience_size, self.batch_size)

        Debug.pause_training_chrono_and_resume_non_training_chrono()

    def copy_target_networks(self):
        """ Make the network that are training do a copy of themselves.
            (refresh TNetworks if the Network is in training mode)
        """
        for network in self.networks:
            if network.is_training:
                network.copy_target_network()

    def train(self, world, nb_episodes, max_score_per_episode=500, stop_on_score_avg=None):
        print("Started training...")

        score_avg = 0
        tmp_total_score = 0

        Debug.start_non_training_chrono()

        for i in range(nb_episodes):
            Debug.print_episode_number(i)
            Debug.manage_chrono_for_begining_of_episode(i)
            Debug.plot_chronos_results(i)

            results = self.play_episode(world, max_score_per_episode)
            tmp_total_score += results['score']

            Debug.write_tensorboard_results(results, self.epsilon.value, self.networks, self.actions_made_histogram, self.actions_made_placeholder, i) # TODO: replace i with Debug.EPISODE_NUMBER ??
            avg_score_printed, score_avg = Debug.print_avg_score(i, tmp_total_score)
            if avg_score_printed:
                tmp_total_score = 0

            if stop_on_score_avg is not None and score_avg is not None and score_avg >= stop_on_score_avg:
                print("Score avg reached. Stop learning")
                self.exit = True

            # exit if we have to
            if self.exit:
                self._save_and_exit()

        print("Nb max episodes reached. Stop learning")
        if Debug.SAVE_FOLDER is not None:
            self._save()
