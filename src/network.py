

class Network:

    def __init__(self, model, input_adapter=None, is_training=False):
        self.is_training = is_training
        self.model = model
        self.input_adapter = input_adapter

        self.prediction_done = False
        self.depends_on = []
        self.experiences = []
        self.last_prediction_values = None # {}
        self.output_placeholder = None

    def add_dependency(network):
        self.depends_on.append(network)

    def predict(self, input=None):
        pass
