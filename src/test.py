from network import *
from world import *
from state import *



model = Model( 10, 1e-2,
    (64, 'relu'),
    (32, 'relu'),
    (Action.NB_POSSIBLE_ACTIONS, 'linear')
)

init = tf.global_variables_initializer()
session = tf.Session()
model.set_session(session)
session.run(init)

writer = tf.summary.FileWriter('../TensorBoard')
writer.add_graph(session.graph)
writer.flush()
