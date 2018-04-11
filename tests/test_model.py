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
            [(64, 'relu'),
            (32, 'relu'),
            (output_dim, 'linear')]
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

    def test_model_train(self):
        input_dim = 10
        output_dim = 2
        model = Model( input_dim, 1e-4,
            [(64, 'relu'),
            (32, 'relu'),
            (output_dim, 'linear')]
        )

        init = tf.global_variables_initializer()
        session = tf.Session()
        model.set_session(session)
        session.run(init)

        # test a training that will change weights
        x = np.array([[0,1,2,3,4,5,6,7,8,9]])
        y_before_training = model.predict(x)
        t = np.array([[1,1]])
        y = [1]
        model.train(x,y,t)
        y_after_training = model.predict(x)
        self.assertFalse(np.array_equal(y_before_training,y_after_training))
        # print(y_before_training)
        # print(y_after_training)

        # test a training that shouldnt change weights
        y_before_training = y_after_training
        t = np.array([[0, y_before_training[0][1]]])
        y = [1]
        self.assertEqual(0.0, model.debug_return_cost(x,y,t))
        # print("Cost: ")
        # print(model.debug_return_cost(x, y,t))
        model.train(x,y,t)
        y_after_training = model.predict(x)
        # print(y_before_training)
        # print(y_after_training)
        self.assertTrue(np.array_equal(y_before_training,y_after_training))


    def test_model_copy(self):
        # create a model and his target but dont copy target from model.
        input_dim = 10
        output_dim = 2
        model = Model( input_dim, 1e-2,
            [(64, 'relu'),
            (32, 'relu'),
            (output_dim, 'linear')]
        )
        target_model = TargetModel(model)

        init = tf.global_variables_initializer()
        session = tf.Session()
        model.set_session(session)
        target_model.set_session(session)
        session.run(init)

        # Their predictions should be different
        x = np.array([[0,1,2,3,4,5,6,7,8,9]])
        y_model = model.predict(x)
        y_target = target_model.predict(x)
        self.assertFalse(np.array_equal(y_model,y_target))

        # now copy the weights from the model to the target. Their predictions should be equals
        x = np.array([[2,4,6,7,6,5,5,7,8,9]])
        target_model.copy_from(model)
        y_model = model.predict(x)
        y_target = target_model.predict(x)
        self.assertTrue(np.array_equal(y_model,y_target))

        # now train the model. The predictions of the model and target should be different this time
        t = np.array([[1,1]])
        y = [1]
        model.train(x,y,t)
        y_model = model.predict(x)
        y_target = target_model.predict(x)
        self.assertFalse(np.array_equal(y_model,y_target))

    def test_model_export_import(self):
        input_dim = 10
        output_dim = 2
        model = Model( input_dim, 1e-2,
            [(64, 'relu'),
            (32, 'relu'),
            (output_dim, 'linear')]
        )

        init = tf.global_variables_initializer()
        session = tf.Session()
        model.set_session(session)
        session.run(init)

        model.export_model('./saves', 'test_model')
