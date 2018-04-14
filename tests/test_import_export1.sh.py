"""
simple script to create a model, get its prediction, and then save the model. 
it will print the predictions. Then run the other script (test_import_export2.sh.py) 
to test that the import imports the same model, and predicts the same exact value. 
"""
import sys
sys.path.append('../src')

from network import *

input_dim = 10
output_dim = 10
model = Model('model1', input_dim, 1e-2,
    [[64, 'relu'],
    [32, 'relu'],
    [output_dim, 'linear']]
)

init = tf.global_variables_initializer()
session = tf.InteractiveSession()
model.set_session(session)
session.run(init)

x = np.array([[12,13,1,2,4,5,6,3,7,3]])
print(model.predict(x))
#model.debug_list_all_variables()

model.export_model('./tmp_test_saves', 'test_import_export_model')


