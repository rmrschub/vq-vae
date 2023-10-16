import tensorflow as tf
import numpy as np


tfkl = tf.keras.layers


@tf.keras.saving.register_keras_serializable()
class ResidualBlock(tf.keras.Model):

    def __init__(self, **kwargs):
    
        super(ResidualBlock, self).__init__()

        # region: Set attributes
        # filter
        # kernel_size
        # stride
        # alpha

        self.__dict__.update(kwargs)

        # endregion

        # Define layers
        self._residual_block = tf.keras.Sequential([
            tfkl.Conv2D(
                filters=self.filters, 
                kernel_size=self.kernel_size,
                strides=self.stride, 
                padding='same'
            ),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(alpha=self.alpha),
            tfkl.Conv2D(
                filters=self.filters, 
                kernel_size=self.kernel_size,
                strides=self.stride, 
                padding='same'
            ),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(alpha=self.alpha),],
        )
        #endregion

    def call(self, inputs):

        return inputs + self._residual_block(inputs)