
class DebugWorld1:

    def step(self, action):
        # action 0 or 1
        if self.first_action is None:
            self.first_action = action
        reward = 0 if action is self.first_action else 1
        return 



    def reset(self):
        self.first_action = None
        self.nb_max_steps = 100
        self.nb_step = 0
