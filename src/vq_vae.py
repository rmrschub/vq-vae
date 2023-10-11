import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from residual_block import ResidualBlock
from vector_quantizer import VectorQuantizer

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


@tf.keras.saving.register_keras_serializable()
class VectorQuantizedVAE(tf.keras.Model):

    def __init__(self, **kwargs):
        super(VectorQuantizedVAE, self).__init__()

        # region: Set attributes
        self.input_dims = kwargs['input_dims']
        self.latent_dim = kwargs['latent_dim']
        self.num_embeddings = kwargs['num_embeddings']
        self.commitment_cost_factor = kwargs['commitment_cost_factor']
        self.quantization_loss_factor = kwargs['quantization_loss_factor']
        self.bernstein_order = kwargs['bernstein_order']
        self.alpha = kwargs['alpha']
        # endregion

        # region Define loss trackers
        self.commitment_loss_tracker = tf.keras.metrics.Mean(name="commitment_loss")
        self.codebook_loss_tracker = tf.keras.metrics.Mean(name="codebook_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # endregion

        self._encoder = tf.keras.Sequential([
                tfkl.Conv2D(filters=256, kernel_size=4, strides=2, padding='same'),
                tfkl.BatchNormalization(),
                tfkl.LeakyReLU(alpha=0.2),
                tfkl.Conv2D(filters=self.latent_dim, kernel_size=4, strides=1, padding='same'),
                ResidualBlock(filters=self.latent_dim, kernel_size=(3, 3), stride=(1, 1), alpha=self.alpha),
                ResidualBlock(filters=self.latent_dim, kernel_size=(3, 3), stride=(1, 1), alpha=self.alpha)],
            name='encoder', 
        )

        self._quantizer = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            latent_dim=self.latent_dim,
        )

        self._decoder = tf.keras.Sequential([
                ResidualBlock(filters=self.latent_dim, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
                ResidualBlock(filters=self.latent_dim, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
                tfkl.Conv2DTranspose(filters=self.latent_dim, kernel_size=4, strides=1, padding='same'),
                tfkl.BatchNormalization(),
                tfkl.LeakyReLU(alpha=0.2),
                tfkl.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same')],
            name='decoder'
        )

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

    def call(self, inputs, training=None, mask=None):

        z_e = self._encoder(inputs)

        codebook_indices, z_q = self._quantizer(z_e)

        p_x_given_z_q = self._likelihood(
            self._decoder(z_e + tf.stop_gradient(z_q - z_e))
        )

        return z_e, codebook_indices, z_q, p_x_given_z_q

    def train_step(self, x):
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            z_e, codebook_indices, z_q, p_x_given_z_q = self(x, training=True)

            commitment_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q)) ** 2)
            codebook_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q) ** 2)
            reconstruction_loss = -1.0 * p_x_given_z_q.log_prob(x) 
            total_loss = sum([
                self.commitment_cost_factor * commitment_loss, 
                self.quantization_loss_factor * codebook_loss, 
                reconstruction_loss
            ])
       
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.commitment_loss_tracker.update_state(commitment_loss)
        self.codebook_loss_tracker.update_state(codebook_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.total_loss_tracker.update_state(total_loss)    

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "commitment_loss": self.commitment_loss_tracker.result(),
            "codebook_loss": self.codebook_loss_tracker.result()
        }
