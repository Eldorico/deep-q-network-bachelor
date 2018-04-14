"""
simple script to import a model and print its prediction
this script should be used after the other script (test_import_export1.sh.py) 
which creates a model and saves it. 
the prediction of this script should be the same as the prediction of the 
test_import_export1.sh.py
"""
import sys
sys.path.append('../src')

from network import *

session = tf.Session()

imported_model = ImportModel(session, './tmp_test_saves', 'test_import_export_model', 'model1')

x = np.array([[12,13,1,2,4,5,6,3,7,3]])
print(imported_model.predict(x))
#imported_model.debug_list_all_variables()

#imported_model.export_model('./saves', 'model2')
