
class Epsilon:
    def __init__(self, start_epsilon_value):
        self.value = start_epsilon_value

    def set_epsilon_function(self, epsilon_function):
        self.epsilon_function = epsilon_function

    def update_epsilon(self):
        self.epsilon_function(self)


class Agent:

    def __init__(self, config, bus):
        self.networks = config['networks']
        self.output_network = config['output_network']
        self.copy_target_period = config['copy_target_period']
        self.min_experience_size = config['min_experience_size']
        self.max_experience_size = config['max_experience_size']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.tensorboard = config['tensorboard'] if 'tensorboard' in config else None

        # # TODO: j'en suis à là
        # https://github.com/keras-team/keras/issues/3358
        if self.tensorboard is not None:
            self.tensorboard.set_model(self.output_network.model)
            self.tensorboard.write_model_graph()

        self.bus = bus

        self.nb_steps_played = 0

    def play_episode(self, world):
        current_state = world.reset()

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

        return world_informations['score']

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
                network.train(self.gamma, self.min_experience_size, self.batch_size, self.tensorboard)

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
        for i in range(nb_episodes):
            print("episode %d" % (i+1))

            score = self.play_episode(world)
            tmp_total_score += score

            if self.tensorboard is not None:
                self.tensorboard.write_summary('score', i, score)
                # self.tensorboard.write_histograms(i)

            if i % avg_every_n_episodes == 0 and i != 0:
                score_avg = tmp_total_score / avg_every_n_episodes
                print("score avg after %d episodes: %f" % (i, score_avg) )
                tmp_total_score = 0
                if stop_on_score_avg is not None and score_avg >= stop_on_score_avg:
                    print("Score avg reached. Stop learning")
                    return

            if i == nb_episodes:
                print("Nb max episodes reached. Stop learning")
                return
