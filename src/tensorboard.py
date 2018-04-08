from keras.callbacks import TensorBoard
from tensorflow.core.framework import summary_pb2
import tensorflow as tf

class Logger(TensorBoard):

    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.writer = tf.summary.FileWriter(self.log_dir)
        # self.log_dir = log_dir

    def write_summary(self, _tag, x_value, y_value):
        value = summary_pb2.Summary.Value(tag=_tag, simple_value=y_value)
        summary = summary_pb2.Summary(value=[value])
        self.writer.add_summary(summary, x_value)
