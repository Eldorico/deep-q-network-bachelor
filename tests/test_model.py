import sys
sys.path.append('../src')

from network import *
from world import *
from state import *
import numpy as np
import unittest

class ModelTest(unittest.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    def test_model_sizes_prediction(self):

        input_dim = 10
        output_dim = 2
        model = Model('test_model', input_dim, 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
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
        model = Model('test_model', input_dim, 1e-4,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
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
        model = Model('test_model', input_dim, 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
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
        """ checks if an model can create a good imported model """
        input_dim = 10
        output_dim = 2
        model = Model('my_model', input_dim, 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
        )

        init = tf.global_variables_initializer()
        session = tf.Session()
        model.set_session(session)
        session.run(init)

        model.export_model('./saves', 'test_model')

        # import model
        imported_model = ImportModel(session, './saves', 'test_model', 'my_model')

        # check for basic non tensorflow attributes
        self.assertEqual(model.learning_rate, imported_model.learning_rate)
        self.assertEqual(model.output_size, imported_model.output_size)
        self.assertEqual(model.name, imported_model.name)
        self.assertTrue(model.args == imported_model.args)

        # check if the wegiths have been correctly restored
        X = np.array([[2,4,6,7,6,5,5,7,8,9]])
        y_model = model.predict(X)
        y_imported_model = imported_model.predict(X)
        self.assertTrue(np.array_equal(y_model,y_imported_model))

    # def test_model_export_import2(self):
    #     """ checks if an imported model can create a good second imported model """
    #     with tf.Session() as sess:
    #         input_dim = 10
    #         output_dim = 2
    #         model = Model( input_dim, 1e-2,
    #             [[64, 'relu'],
    #             [32, 'relu'],
    #             [output_dim, 'linear']]
    #         )
    #         init = tf.global_variables_initializer()
    #         sess.run(init)
    #         model.set_session(sess)
    #         model.export_model('./saves', 'test_model')
    #         #
    #
    #     # init = tf.global_variables_initializer()
    #     # session = tf.Session()
    #     # model.set_session(session)
    #     # session.run(init)
    #
    #     # create an imported model, and then export the imported model
    #     tf.reset_default_graph()
    #
    #
    #     with tf.Session() as sess:
    #         init = tf.global_variables_initializer()
    #         sess.run(init)
    #         # tf.initialize_all_variables().run()
    #         imported_model = ImportModel(sess, './saves', 'test_model')
    #         imported_model.export_model('./saves', 'test_imported_model')
    #     #
    #     # # import a model from the imported model
    #     # imported_model2 = ImportModel(session, './saves', 'test_imported_model')
    #     #
    #     # # check for basic non tensorflow attributes
    #     # self.assertEqual(imported_model.learning_rate, imported_model2.learning_rate)
    #     # self.assertEqual(imported_model.output_size, imported_model2.output_size)
    #     # self.assertEqual(imported_model.name, imported_model2.name)
    #     # self.assertTrue(imported_model.args == imported_model2.args)
