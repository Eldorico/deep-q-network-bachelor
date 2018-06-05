import time
import datetime
import shutil
import __main__

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
        return Global.SAVE_FOLDER + '/TB_'+ datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

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
