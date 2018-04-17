import os
import sys
sys.path.append('../src')

from network import *
from world import *
from state import *
import numpy as np
import unittest


class ModelTest(unittest.TestCase):

    def __init__(self, args):
        super().__init__(args)
        self.save_folder = './tmp_test_saves'

    def remove_all_test_save_files(self):
        files_to_remove = [ f for f in os.listdir(self.save_folder)]
        for f in files_to_remove:
            os.remove(os.path.join(self.save_folder, f))

    def close_session_and_reset_default_graph(self, session):
        tf.reset_default_graph()
        session.close()

    def print_checkpoint_variables(self, base_path):
        from tensorflow.python import pywrap_tensorflow
        reader = pywrap_tensorflow.NewCheckpointReader(base_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        print(var_to_shape_map)

    def check_non_tensorflow_attributes_equality(self, model1, model2):
        self.assertEqual(model1.learning_rate, model2.learning_rate)
        self.assertEqual(model1.output_size, model2.output_size)
        self.assertEqual(model1.name, model2.name)
        self.assertTrue(model1.args == model2.args)

    def test_model_sizes_prediction(self):

        session = tf.Session()

        input_dim = 10
        output_dim = 2
        model = Model(session, 'test_model', input_dim, 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
        )

        x = np.array([[0,1,2,3,4,5,6,7,8,9]])
        y = model.predict(x)
        self.assertEqual((1,output_dim), y.shape)

        x = np.array([[0,1,2,3,4,5,6,7,8,9], [0,1,2,3,3,54,7,1,9,10], [9,8,7,6,5,4,3,2,1,0]])
        y = model.predict(x)
        self.assertEqual((3,output_dim), y.shape)

    def test_model_train(self):
        session = tf.Session()

        input_dim = 10
        output_dim = 2
        model = Model(session, 'test_model', input_dim, 1e-4,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
        )

        # test a training that will change weights
        x = np.array([[0,1,2,3,4,5,6,7,8,9]])
        y_before_training = model.predict(x)
        t = np.array([[1,1]])
        y = [1]
        model.train(x,y,t)
        y_after_training = model.predict(x)
        self.assertFalse(np.array_equal(y_before_training,y_after_training))

        # test a training that shouldnt change weights
        y_before_training = y_after_training
        t = np.array([[0, y_before_training[0][1]]])
        y = [1]
        self.assertEqual(0.0, model.debug_return_cost(x,y,t))
        model.train(x,y,t)
        y_after_training = model.predict(x)
        self.assertTrue(np.array_equal(y_before_training,y_after_training))

    def test_model_copy(self):
        session = tf.Session()

        # create a model and his target but dont copy target from model.
        input_dim = 10
        output_dim = 2
        model = Model(session, 'test_model', input_dim, 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
        )
        target_model = TargetModel(model)

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
        tf.reset_default_graph()
        session = tf.Session()

        # create the base model
        input_dim = 10
        output_dim = 2
        model = Model(session, 'my_model', input_dim, 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
        )

        # export the base model
        model.export_model(self.save_folder)

        # get the model variables to test before closing the session
        X = np.array([[2,4,6,7,6,5,5,7,8,9]])
        y_model = model.predict(X)  # simple prediction
        t = np.array([[0, 1]])
        y = [1]
        model.train(X, y, t) # a train that should change the variables
        y_model_after_train = model.predict(X)

        # close the session and reset the graph
        self.close_session_and_reset_default_graph(session)

        # import model
        session = tf.Session()
        imported_model = ImportModel(session, self.save_folder, 'my_model')

        # check for basic non tensorflow attributes
        self.check_non_tensorflow_attributes_equality(model, imported_model)

        # check if the wegiths have been correctly restored
        y_imported_model = imported_model.predict(X)
        self.assertTrue(np.array_equal(y_model,y_imported_model))
        self.assertFalse(np.array_equal(y_model_after_train,y_imported_model))

    def test_model_export_import2(self):
        """ checks if an imported model can create a good second imported model """
        tf.reset_default_graph()
        session = tf.Session()

        # create the base model
        input_dim = 10
        output_dim = 12
        model1 = Model(session, 'model', input_dim, 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
        )

        # export the base model
        model1.export_model(self.save_folder)

        # get the model variables to test before closing the session
        X = np.array([[2,4,6,7,6,5,5,7,8,9]])
        y_model1 = model1.predict(X)  # simple prediction

        # close the session and reset the graph
        self.close_session_and_reset_default_graph(session)

        # create the model2, remove all saves files and export it
        session = tf.Session()
        model2 = ImportModel(session, self.save_folder, 'model')

        # remove all save files of the tests
        self.remove_all_test_save_files()

        # export model2
        model2.export_model(self.save_folder)

        # close the session and reset the graph
        self.close_session_and_reset_default_graph(session)

        # create the model3
        session = tf.Session()
        model3 = ImportModel(session, self.save_folder, 'model')

        # check for basic non tensorflow attributes
        self.check_non_tensorflow_attributes_equality(model1, model3)

        # check if the wegiths have been correctly restored
        y_model3 = model3.predict(X)
        self.assertTrue(np.array_equal(y_model1, y_model3))

    def test_model_export_import3(self):
        """ Tests the case where multiple models have been instanciated and exported.
            But we only want to restore one particular model.
        """
        tf.reset_default_graph()
        session = tf.Session()

        # create the model1
        input_dim = 10
        output_dim = 12
        model1 = Model(session, 'model1', input_dim, 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
        )

        # create the model2
        input_dim = 12
        output_dim = 13
        model2 = Model(session, 'model2', input_dim, 1e-2,
            [[64, 'relu'],
            [32, 'relu'],
            [output_dim, 'linear']]
        )

        # export each model
        model1.export_model(self.save_folder)
        model2.export_model(self.save_folder)

        # test prediction
        y_1 = model1.predict([[1,3,2,4,5,2,6,3,4,6]])
        y_2 = model2.predict([[2,3,6,4,32,6,2,67,4,3,12,6]])

        # reset graph and close session
        self.close_session_and_reset_default_graph(session)

        # debug
        # self.print_checkpoint_variables(self.save_folder+'/import_test3_model1')
        # print("ASDFASDFASDF")
        # self.print_checkpoint_variables(self.save_folder+'/import_test3_model2')

        # import only one model (model2)
        session = tf.Session()
        model2_bis = ImportModel(session, self.save_folder, 'model2')

        # check prediction of model2
        y_2_bis = model2_bis.predict([[2,3,6,4,32,6,2,67,4,3,12,6]])
        self.assertTrue(np.array_equal(y_2, y_2_bis))

        # check if we have been here without error
        self.assertTrue(True)
