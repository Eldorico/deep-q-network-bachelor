import sys
sys.path.append('../src')

from network import *
from world import *
from state import *
import numpy as np
import unittest

class ModelTest(unittest.TestCase):

    def test_model_sizes_prediction(self):

        input_dim = 10
        output_dim = 2
        model = Model( input_dim, 1e-2,
            (64, 'relu'),
            (32, 'relu'),
            (output_dim, 'linear')
        )

        init = tf.global_variables_initializer()
        session = tf.Session()
        model.set_session(session)
        session.run(init)

        x = np.array([[0,1,2,3,4,5,6,7,8,9]])
        y = model.predict(x)
        self.assertEqual((1,output_dim), y.shape)

        x = np.array([[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,3,54,7,1,9,10], [9,8,7,6,5,4,3,2,1,0]])
        y = model.predict(x)
        self.assertEqual((3,output_dim), y.shape)
