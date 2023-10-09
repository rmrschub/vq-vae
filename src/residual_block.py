import tensorflow as tf
import numpy as np


tfkl = tf.keras.layers


@tf.keras.saving.register_keras_serializable()
class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
    
        super(ResidualBlock, self).__init__()

        # region: Set attributes
        self.filters = kwargs['filters']
        self.kernel_size= kwargs['kernel_size']    # (3, 3)
        self.stride = kwargs['stride']             # (1, 1)
        self.alpha = kwargs['alpha']               # 0.2
        # endregion

        # region: Define layers
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

    def call(self, x, **kwargs):

        return x + self._residual_block(x)