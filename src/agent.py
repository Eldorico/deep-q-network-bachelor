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

        # if Global.USE_TENSORBOARD:
        #     Global.EPISODE_NUMBER = 0
        #
        #     # TODO: remove the self.writer ?
        #     self.writer = tf.summary.FileWriter(Global.get_TB_folder())
        #     self.writer.add_graph(Global.SESSION.graph)
        #     Global.WRITER = self.writer
        #
        #     self.actions_made_placeholder = tf.placeholder(tf.int32, [None, 1])
        #     self.actions_made_histogram = tf.summary.histogram('actions_distribution', self.actions_made_placeholder)

        # save the main file before someone change it by error
        # if Global.SAVE_FOLDER is not None and Global.SAVE_MAIN_FILE:
        #     print("Saving mainfile in save folder: %s..." %Global.SAVE_FOLDER)
        #     shutil.copyfile(__main__.__file__, Global.SAVE_FOLDER + '/' + 'main_file.py')
        #     print("Done")

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

        self.bus['last_states'] = [current_state for i in range(4)]

        #debug
        # debug = 0

        episode_done = False
        while not episode_done:

            # debug
            # print("\nStep %d:" % debug)

            action = self.choose_action(current_state)
            next_state, reward, game_over, world_debug = world.step(action)

            # debug
            # debug += 1
            # if debug >= 10:
            #     exit()
            # else:
            #     print("bus['last_states'] = ")
            #     print(self.bus['last_states'])

            # debug
            # if world_debug['score'] >= 50:
            #     print("episode %d: score = %d" % (Global.EPISODE_NUMBER, world_debug['score']))

            # if Global.PRINT_REWARD_EVERY_N_EPISODES > 0 and Global.EPISODE_NUMBER % Global.PRINT_REWARD_EVERY_N_EPISODES == 0:
            #     print("Ennemy.x=%d, Ennemy.y=%d, Ennemy.direction=%s, agent.x=%d, agent.y=%d, action=%d Reward: %f" % ( world_debug['ennemies_position'][0][0],
            #                                                                                                             world_debug['ennemies_position'][0][1],
            #                                                                                                             Direction.toStr[world_debug['ennemies_position'][0][2]],
            #                                                                                                             world_debug['agent_x'],
            #                                                                                                             world_debug['agent_y'],
            #                                                                                                             action,
            #                                                                                                             reward))
                # debug
                # print("Debug: current ennemy.x=%d, ennemy.y=%d, direction=%d" %(world.ennemies[0].x, world.ennemies[0].y, world.ennemies[0].direction))

            if self.nb_steps_played % self.copy_target_period == 0:
                self.copy_target_networks()
                Debug.print_target_network_copied()

                # if Global.SAY_WHEN_TARGET_NETWORK_COPIED:
                #     print("Target Networks copied")

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
        # Global.EPISODE_NUMBER += 1
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

        # debug
        # print("choose_action(): bus['state'] = %s" % self.bus['state'])
        # print(self.bus['state'].get_ennemy_agent_layer_only())

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

        #debug
        # print("add_experience (agent): bus['next_state'] = %s" % self.bus['next_state'])
        # print(self.bus['next_state'])
        # print("bus['last_states'] = ")
        # print(self.bus['last_states'])

        for network in self.networks:
            if network.is_training:
                network.add_experience(self.bus, reward, game_over, self.max_experience_size, world)

    def train_networks(self):
        Debug.pause_non_training_chrono_and_resume_training_chrono()
        # if Global.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES is not 0:
        #     chrono = Chronometer()
        #     chrono.pause_chrono(Chronometer.NON_TRAINING_CHRONO)
        #     chrono.resume_chrono(Chronometer.TRAINING_CHRONO)

        for network in self.networks:
            if network.is_training:
                network.train(self.gamma, self.min_experience_size, self.batch_size)

        Debug.pause_training_chrono_and_resume_non_training_chrono()
        # if Global.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES is not 0:
        #     chrono = Chronometer()
        #     chrono.pause_chrono(Chronometer.TRAINING_CHRONO)
        #     chrono.resume_chrono(Chronometer.NON_TRAINING_CHRONO)

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

        # if Debug.USE_TENSORBOARD:
        #     log = {'actions_made' : []}

        Debug.start_non_training_chrono()
        # if Global.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES is not 0:
        #     chrono = Chronometer()
        #     chrono.resume_chrono(Chronometer.NON_TRAINING_CHRONO)

        for i in range(nb_episodes):
            Debug.print_episode_number(i)
            # if Global.PRINT_EPISODE_NB_EVERY_N_EPISODES > 0 and i != 0 and i % Global.PRINT_EPISODE_NB_EVERY_N_EPISODES == 0:
            #     print("episode %d" % (i))

            Debug.manage_chrono_for_begining_of_episode(i)
            # if Global.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES is not 0 and i % Global.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES == 0:
            #     chrono.pause_chrono(Chronometer.NON_TRAINING_CHRONO)
            #     chrono.checkpoint_chrono(Chronometer.TRAINING_CHRONO, Global.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES)
            #     chrono.checkpoint_chrono(Chronometer.NON_TRAINING_CHRONO, Global.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES)
            #     chrono.resume_chrono(Chronometer.NON_TRAINING_CHRONO)
            Debug.plot_chronos_results(i)
            # if Global.PLOT_TIMES_DURATION_ON_N_EPISODES is not 0 and Global.PLOT_TIMES_DURATION_ON_N_EPISODES == i:
            #     print("about to plot")
            #     chrono.plot_chrono_deltas()
            #     print("plotted")

            results = self.play_episode(world, max_score_per_episode)
            tmp_total_score += results['score']

            # debug
            # if results['score'] > 100:
            #     print('score: %f' %results['score'])

            # debug
            # if results['score'] >= 20:
            #     print("agent.train(): score greater than 20: episode = %d, score = %d" % (i, results['score']))

            # update tensorboard
            Debug.write_tensorboard_results(results, self.epsilon.value, self.networks, self.actions_made_histogram, self.actions_made_placeholder, i) # TODO: replace i with Debug.EPISODE_NUMBER ??
            # if Global.USE_TENSORBOARD and Global.EPISODE_NUMBER % Global.OUTPUT_TO_TENSORBOARD_EVERY_N_EPISODES == 0:
            #     value = summary_pb2.Summary.Value(tag="score_per_episode", simple_value=results['score'])
            #     summary = summary_pb2.Summary(value=[value])
            #     self.writer.add_summary(summary, i)
            #
            #     value = summary_pb2.Summary.Value(tag="epsilon_value", simple_value=self.epsilon.value)
            #     summary = summary_pb2.Summary(value=[value])
            #     self.writer.add_summary(summary, i)
            #
            #     value = summary_pb2.Summary.Value(tag="total reward per episode", simple_value=results['total_reward'])
            #     summary = summary_pb2.Summary(value=[value])
            #     self.writer.add_summary(summary, i)
            #
            #     log['actions_made'] += results['actions_made']
            #     # if i % 50 == 0:  # TODO: reput 50 instead of 10!
            #     summary = Global.SESSION.run(self.actions_made_histogram, feed_dict={self.actions_made_placeholder: np.reshape(log['actions_made'], (len(log['actions_made']), 1))})
            #     self.writer.add_summary(summary, i)
            #     log['actions_made'] = []
            #
            #     for network in self.networks:
            #         if network.is_training:
            #             network.model.write_weights_tb_histograms()
            #             if Global.SAY_WHEN_HISTOGRAMS_ARE_PRINTED:
            #                 print("weights histograms printed")


            avg_score_printed, score_avg = Debug.print_avg_score(i, tmp_total_score)
            if avg_score_printed:
                tmp_total_score = 0
            if stop_on_score_avg is not None and score_avg is not None and score_avg >= stop_on_score_avg:
                print("Score avg reached. Stop learning")
                self.exit = True
            # if Global.PRINT_SCORE_AVG_EVERY_N_EPISODES > 0 and i % Global.PRINT_SCORE_AVG_EVERY_N_EPISODES == 0 and i != 0:
            #     score_avg = tmp_total_score / Global.PRINT_SCORE_AVG_EVERY_N_EPISODES
            #     print("score avg after %d episodes: %f" % (i, score_avg) )
            #     tmp_total_score = 0
            # check if avg score is reached
            # if stop_on_score_avg is not None and score_avg >= stop_on_score_avg:
            #     print("Score avg reached. Stop learning")
            #     self.exit = True

            # exit
            if self.exit:
                self._save_and_exit()

        print("Nb max episodes reached. Stop learning")
        if Debug.SAVE_FOLDER is not None:
            self._save()
