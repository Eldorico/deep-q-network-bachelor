import time
import datetime
import shutil
import __main__
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

from state import *
from chronometer import *

class Debug:
    USE_TENSORBOARD = False
    SAVE_MAIN_FILE = False
    SAVE_FOLDER = None
    SESSION = None
    WRITER = None
    EPISODE_NUMBER = -1
    PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES = 0
    PRINT_PREDICTED_VALUES_FOR = []
    PRINT_REWARD_EVERY_N_EPISODES = 0
    PRINT_EPISODE_NB_EVERY_N_EPISODES = 1
    PRINT_SCORE_AVG_EVERY_N_EPISODES = 1
    PRINT_TARGET_COPY_RATIO = False
    _NB_TRAINED_STEPS = 0
    SAY_WHEN_TARGET_NETWORK_COPIED = False
    SAY_WHEN_AGENT_TRAINED = False
    SAY_WHEN_HISTOGRAMS_ARE_PRINTED = False
    OUTPUT_TO_TENSORBOARD_EVERY_N_EPISODES = 100
    PLOT_TIMES_DURATION_ON_N_EPISODES = 0
    RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES = 0

    _actions_made_placeholder = None
    _actions_made_histogram = None

    @staticmethod
    def get_TB_folder():
        return Debug.SAVE_FOLDER + '/TB_'+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    @staticmethod
    def set_episode_number(episode_number):
        Debug.EPISODE_NUMBER = 0

    @staticmethod
    def set_session(session):
        Debug.SESSION = session

    @staticmethod
    def create_tensorboard_writer():
        if Debug.USE_TENSORBOARD:
            Debug.WRITER = tf.summary.FileWriter(Debug.get_TB_folder())
            Debug.WRITER.add_graph(Debug.SESSION)

            Debug._actions_made_placeholder = tf.placeholder(tf.int32, [None, 1])
            Debug._actions_made_histogram = tf.summary.histogram('actions_distribution', Debug._actions_made_placeholder)
            return Debug._actions_made_placeholder, Debug._actions_made_histogram # TODO: why should I return this now?

    @staticmethod
    def save_main_file():
        if Debug.SAVE_FOLDER is not None and Debug.SAVE_MAIN_FILE:
            print("Saving mainfile in save folder: %s..." %Debug.SAVE_FOLDER)
            shutil.copyfile(__main__.__file__, Debug.SAVE_FOLDER + '/' + 'main_file.py')
            print("Done")

    @staticmethod
    def print_reward(world_debug, action, reward):
        if Debug.PRINT_REWARD_EVERY_N_EPISODES > 0 and Debug.EPISODE_NUMBER % Debug.PRINT_REWARD_EVERY_N_EPISODES == 0:
            print("Ennemy.x=%d, Ennemy.y=%d, Ennemy.direction=%s, agent.x=%d, agent.y=%d, action=%d Reward: %f" % ( world_debug['ennemies_position'][0][0],
                                                                                                                    world_debug['ennemies_position'][0][1],
                                                                                                                    Direction.toStr[world_debug['ennemies_position'][0][2]],
                                                                                                                    world_debug['agent_x'],
                                                                                                                    world_debug['agent_y'],
                                                                                                                    action,
                                                                                                                    reward))


    @staticmethod
    def print_target_network_copied():
        if Debug.SAY_WHEN_TARGET_NETWORK_COPIED:
            print("Target Networks copied")

    @staticmethod
    def increment_episode_number():
        Debug.EPISODE_NUMBER += 1

    @staticmethod
    def pause_non_training_chrono_and_resume_training_chrono():
        if Debug.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES is not 0:
            chrono = Chronometer()
            chrono.pause_chrono(Chronometer.NON_TRAINING_CHRONO)
            chrono.resume_chrono(Chronometer.TRAINING_CHRONO)

    @staticmethod
    def pause_training_chrono_and_resume_non_training_chrono():
        if Debug.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES is not 0:
            chrono = Chronometer()
            chrono.pause_chrono(Chronometer.TRAINING_CHRONO)
            chrono.resume_chrono(Chronometer.NON_TRAINING_CHRONO)

    @staticmethod
    def start_non_training_chrono():
        if Debug.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES is not 0:
            chrono = Chronometer()
            chrono.resume_chrono(Chronometer.NON_TRAINING_CHRONO)

    @staticmethod
    def print_episode_number(episode_number):
        if Debug.PRINT_EPISODE_NB_EVERY_N_EPISODES > 0 and episode_number != 0 and episode_number % Debug.PRINT_EPISODE_NB_EVERY_N_EPISODES == 0:
            print("episode %d" % episode_number)

    @staticmethod
    def manage_chrono_for_begining_of_episode(episode_number):
        if Debug.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES is not 0 and episode_number % Debug.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES == 0:
            chrono = Chronometer()
            chrono.pause_chrono(Chronometer.NON_TRAINING_CHRONO)
            chrono.checkpoint_chrono(Chronometer.TRAINING_CHRONO, Debug.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES)
            chrono.checkpoint_chrono(Chronometer.NON_TRAINING_CHRONO, Debug.RECORD_EVERY_TIME_DURATION_EVERY_N_EPISODES)
            chrono.resume_chrono(Chronometer.NON_TRAINING_CHRONO)

    @staticmethod
    def plot_chronos_results(episode_number):
        if Debug.PLOT_TIMES_DURATION_ON_N_EPISODES is not 0 and Debug.PLOT_TIMES_DURATION_ON_N_EPISODES == episode_number:
            print("about to plot")
            chrono = Chronometer()
            chrono.plot_chrono_deltas()
            print("plotted")

    @staticmethod
    def write_tensorboard_results(results, epsilon_value, networks, episode_number): # TODO: replace episode_number with Debug.EPISODE_NUMBER ??
        if Debug.USE_TENSORBOARD and Debug.EPISODE_NUMBER % Debug.OUTPUT_TO_TENSORBOARD_EVERY_N_EPISODES == 0:
            value = summary_pb2.Summary.Value(tag="score_per_episode", simple_value=results['score'])
            summary = summary_pb2.Summary(value=[value])
            Debug.WRITER.add_summary(summary, episode_number)

            value = summary_pb2.Summary.Value(tag="epsilon_value", simple_value=epsilon_value)
            summary = summary_pb2.Summary(value=[value])
            Debug.WRITER.add_summary(summary, episode_number)

            value = summary_pb2.Summary.Value(tag="total reward per episode", simple_value=results['total_reward'])
            summary = summary_pb2.Summary(value=[value])
            Debug.WRITER.add_summary(summary, episode_number)

            # log['actions_made'] += results['actions_made']
            summary = Debug.SESSION.run(Debug.WRITER._actions_made_histogram, feed_dict={Debug.WRITER._actions_made_placeholder: np.reshape(results['actions_made'], (len(results['actions_made']), 1))}) # TODO: why reshape results['actions_made']? maybe it isnt useful anymore
            Debug.WRITER.add_summary(summary, episode_number)
            # log['actions_made'] = []

            for network in networks:
                if network.is_training:
                    network.model.write_weights_tb_histograms()
                    if Debug.SAY_WHEN_HISTOGRAMS_ARE_PRINTED:
                        print("weights histograms printed")
    @staticmethod
    def print_avg_score(episode_number, total_score): # TODO: replace episode_number by Debug.EPISODE_NUMBER ?
        if Debug.PRINT_SCORE_AVG_EVERY_N_EPISODES > 0 and episode_number % Debug.PRINT_SCORE_AVG_EVERY_N_EPISODES == 0 and episode_number != 0:
            score_avg = tmp_total_score / Debug.PRINT_SCORE_AVG_EVERY_N_EPISODES
            print("score avg after %d episodes: %f" % (episode_number, score_avg) )
            return True, score_avg
        return False, None

    @staticmethod
    def print_predicted_values(name, predicted_values):
        if (len(Debug.PRINT_PREDICTED_VALUES_FOR) == 0 or name in Debug.PRINT_PREDICTED_VALUES_FOR) and Debug.PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES > 0 and Debug.EPISODE_NUMBER % Debug.PRINT_PREDICTED_VALUES_ON_EVERY_N_EPISODES == 0:
            max_index = np.argmax(predicted_values[0])
            max_value = predicted_values[0][max_index]
            min_index = np.argmin(predicted_values[0])
            min_value = predicted_values[0][min_index]
            print("predicted values: max: %d -> %f, min: %d -> %f" % (max_index, max_value, min_index, min_value))

    @staticmethod
    def is_time_to_write_summary():
        return Debug.USE_TENSORBOARD and Debug.EPISODE_NUMBER % Debug.OUTPUT_TO_TENSORBOARD_EVERY_N_EPISODES == 0:

    @staticmethod
    def write_summary(summary_op):
        summary = Debug.SESSION.run(summary_op)
        Debug.WRITER.add_summary(summary, Debug.EPISODE_NUMBER)

    @staticmethod
    def print_target_network_copied():
        if Debug.PRINT_TARGET_COPY_RATIO:
            print("Target network copied after having trained %d steps" % Debug._NB_TRAINED_STEPS)
            Debug._NB_TRAINED_STEPS = 0

    @staticmethod
    def print_network_has_trained():
        if Debug.SAY_WHEN_AGENT_TRAINED:
            print("Network trained")

    @staticmethod
    def increment_nb_training_steps(increment):
        if Debug.PRINT_TARGET_COPY_RATIO:
            Debug._NB_TRAINED_STEPS += increment
