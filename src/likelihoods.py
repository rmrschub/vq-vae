import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from residual_block import ResidualBlock
from vector_quantizer import VectorQuantizer

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


@tf.keras.saving.register_keras_serializable()
class BernsteinLikelihood(tf.keras.Model):

    def __init__(self, **kwargs):
        super(BernsteinLikelihood, self).__init__()

        self.bernstein_order = kwargs['bernstein_order']

        self._likelihood = tf.keras.Sequential([
                tfkl.Conv2DTranspose(
                    filters=3 * (self.bernstein_order + 1), 
                    kernel_size=4, 
                    strides=1, 
                    padding='same', 
                    activation=None),      
                tfkl.Reshape(
                    [32, 32, 3, self.bernstein_order + 1, ],
                    name='bernstein_polynomial_likelihood_params'
                ),
                tfpl.DistributionLambda(
                    make_distribution_fn=lambda t: tfd.Independent(
                        distribution=tfd.MixtureSameFamily(
                            mixture_distribution=tfd.Categorical(
                                logits=t,
                                validate_args=False,
                                allow_nan_stats=False,
                            ),
                            components_distribution=tfd.Kumaraswamy(
                                concentration1=tf.range(0, self.bernstein_order + 1, delta=1, dtype=float) + 1.0,
                                concentration0=self.bernstein_order - tf.range(0, self.bernstein_order + 1, delta=1, dtype=float) + 1.0,
                                validate_args=False,
                                allow_nan_stats=False,
                            ),
                            validate_args=False,
                            allow_nan_stats=False),
                        reinterpreted_batch_ndims=3,
                        validate_args=False,
                        experimental_use_kahan_sum=False),
                    name='bernstein_polynomial_likelihood')],
            name='likelihood', 
        )


    def call(self, inputs):

        return self._likelihood(inputs)