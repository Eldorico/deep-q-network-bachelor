

class Agent:

    def __init__(self, config, bus):
        self.networks = config['networks']
        self.output_network = config['output_network']
        self.copy_target_period = config['copy_target_period']
        self.min_experience_size = config['min_experience_size']
        self.max_experience_size = config['max_experience_size']
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']

        self.bus = bus

        self.nb_steps_played = 0

    def play_episode(self, world):
        current_state = world.reset()

        while not world.game_over:
            action = self.choose_action(current_state)
            next_state, reward, game_over, _ = world.step(action)

            if self.nb_steps_played % self.copy_target_period == 0:
                self.copy_target_networks()

            self.add_experience(next_state, reward)
            self.flush_last_prediction_var()
            self.train_networks()

            current_state = next_state

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
                    network.predict(self.bus, self.epsilon)
                    nb_networks_that_predicted += 1

        return self.output_network.last_prediction_values['action']

    def add_experience(self, next_state, reward, game_over):
        """ add experiences to the networks that are training
        """
        for network in self.networks:
            if network.is_training:
                network.add_experience(next_state, reward, game_over, self.max_experience_size)

    def train_networks(self):
        for network in self.networks:
            if network.is_training:
                network.train_network(self.gamma, self.min_experience_size, self.batch_size)

    def copy_target_networks(self):
        """ Make the network that are training do a copy of themselves.
            (refresh TNetworks if the Network is in training mode)
        """
        for network in self.networks:
            if network.is_training:
                network.copy_target_network()
