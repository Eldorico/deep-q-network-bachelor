from keras.callbacks import TensorBoard
from tensorflow.core.framework import summary_pb2
import tensorflow as tf

class Logger(TensorBoard):

    def __init__(self, log_dir):
        super().__init__(log_dir, histogram_freq=1)
        # self.writer = tf.summary.FileWriter(self.log_dir)
        # self.log_dir = log_dir

    def write_summary(self, _tag, x_value, y_value):
        if hasattr(self, 'writer') and self.writer is not None:
            value = summary_pb2.Summary.Value(tag=_tag, simple_value=y_value)
            summary = summary_pb2.Summary(value=[value])
            self.writer.add_summary(summary, x_value)

    def on_train_end(self, _):
        pass # dont close the writer since the model will be trained after each episode

    def write_model_graph(self):
        self.writer.add_graph(self.sess.graph)

    # def on_epoch_end(self, epoch, logs):
    #     """ patch from https://github.com/keras-team/keras/issues/3358 in order to
    #         write histograms on tensorboard
    #     """
    #     # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
    #     # Below is an example that yields images and classification tags.
    #     # After it's filled in, the regular on_epoch_end method has access to the validation_data.
    #     imgs, tags = None, None
    #     for s in range(1):
    #         ib, tb = next(self.batch_gen)
    #         if imgs is None and tags is None:
    #             imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
    #             tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
    #         imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
    #         tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
    #     self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
    #     return super().on_epoch_end(epoch, logs)

    def write_histograms(self, x_value):
        pass



        # if hasattr(self, 'writer') and self.writer is not None:
        #         val_data = self.validation_data
        #         tensors = (self.model.inputs +
        #                    self.model.targets +
        #                    self.model.sample_weights)
        #
        #         if self.model.uses_learning_phase:
        #             tensors += [K.learning_phase()]
        #
        #         assert len(val_data) == len(tensors)
        #         val_size = val_data[0].shape[0]
        #         i = 0
        #         while i < val_size:
        #             step = min(self.batch_size, val_size - i)
        #             if self.model.uses_learning_phase:
        #                 # do not slice the learning phase
        #                 batch_val = [x[i:i + step] for x in val_data[:-1]]
        #                 batch_val.append(val_data[-1])
        #             else:
        #                 batch_val = [x[i:i + step] for x in val_data]
        #             assert len(batch_val) == len(tensors)
        #             feed_dict = dict(zip(tensors, batch_val))
        #             result = self.sess.run([self.merged], feed_dict=feed_dict)
        #             summary_str = result[0]
        #             self.writer.add_summary(summary_str, x_value)
        #             i += self.batch_size
