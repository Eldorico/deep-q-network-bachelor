from network import *
from world import *
from state import *
import numpy as np

# input_dim = 10
# output_dim = 2
# model = Model( input_dim, 1e-2,
#     [(64, 'relu'),
#     (32, 'relu'),
#     (output_dim, 'linear')]
# )
#
# init = tf.global_variables_initializer()
# session = tf.Session()
# model.set_session(session)
# session.run(init)
#
# x = np.array([[0,1,2,3,4,5,6,7,8,9]])
# y = model.predict(x)

model = ImportModel("asdf", "asdf")
