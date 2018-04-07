

class Agent:

    def __init__(self, config):
        self.networks = config['networks']
        self.last_network = config['last_network']
        self.copy_target_period = config['copy_target_period']

        self.nb_steps_played = 0

    def play_episode(self, world):
        current_state = world.reset()

        while not world.game_over:
            action = self.choose_action(current_state)
            next_state, reward, _ = world.step(action)

            if self.nb_steps_played % self.copy_target_period == 0:
                self.copy_target_networks()

            self.add_experience(next_state, reward)
            self.train_networks()

            current_state = next_state

    def choose_action(state):
        pass

    def predict_q(self, state):
        nb_networks_that_predicted = 0
        while nb_networks_that_predicted != len(self.networks):
            for network in self.networks:
                if len(network.depends_on) is 0:
                    network.predict(state)
                    nb_networks_that_predicted += 1
                elif all(dependency.prediction_done for dependency in network.depends_on):
                    network.predict()
                    nb_networks_that_predicted += 1

        return self.last_network.output_placeholder

    def add_experience(self, next_state, reward):
        pass

    def train_networks(self):
        pass

    def copy_target_networks(self):
        """ Make the network that are training do a copy of themselves.
            (refresh TNetworks if the Network is in training mode)
        """
