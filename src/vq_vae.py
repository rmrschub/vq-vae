import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from residual_block import ResidualBlock
from vector_quantizer import VectorQuantizer
from likelihoods import BernsteinLikelihood

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
        self.random_seed = kwargs['random_seed']
        # endregion

        # region Define loss trackers
        self.commitment_loss_tracker = tf.keras.metrics.Mean(name="commitment_loss")
        self.codebook_loss_tracker = tf.keras.metrics.Mean(name="codebook_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # endregion

        self._encoder = tf.keras.Sequential([
                tfkl.Conv2D(filters=256, kernel_size=4, strides=2, padding='same'),
                # tfkl.BatchNormalization(),
                # tfkl.LeakyReLU(alpha=0.2),
                tfkl.Conv2D(filters=256, kernel_size=4, strides=2, padding='same'),
                ResidualBlock(filters=256, kernel_size=(3, 3), stride=(1, 1), alpha=self.alpha),
                ResidualBlock(filters=256, kernel_size=(3, 3), stride=(1, 1), alpha=self.alpha),
                tfkl.Conv2D(filters=self.latent_dim, kernel_size=1, strides=1, padding='same')],
            name='encoder', 
        )

        self._quantizer = VectorQuantizer(
            num_embeddings=self.num_embeddings,
            latent_dim=self.latent_dim,
            random_seed=self.random_seed
        )

        self._decoder = tf.keras.Sequential([
                tfkl.Conv2D(filters=256, kernel_size=1, strides=1, padding='same'),
                ResidualBlock(filters=256, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
                ResidualBlock(filters=256, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
                tfkl.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same'),
                # tfkl.BatchNormalization(),
                # tfkl.LeakyReLU(alpha=0.2),
                tfkl.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same'),
                tfkl.Conv2DTranspose(filters=3, kernel_size=4, strides=1, padding='same')],
                # BernsteinLikelihood(bernstein_order=self.bernstein_order)],
            name='decoder'
        )

    def call(self, inputs, training=None, mask=None):

        z_e = self._encoder(inputs)

        codebook_indices, z_q = self._quantizer(z_e)

        p_x_given_z_q = self._decoder(z_e + tf.stop_gradient(z_q - z_e))

        return z_e, codebook_indices, z_q, p_x_given_z_q

    def train_step(self, x):
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            z_e, codebook_indices, z_q, p_x_given_z_q = self(x, training=True)

            # commitment_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q)) ** 2)
            # codebook_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q) ** 2)
            # reconstruction_loss = -1.0 * p_x_given_z_q.log_prob(x) 
            commitment_loss = tf.reduce_mean(tf.math.square(z_e - tf.stop_gradient(z_q)))
            codebook_loss = tf.reduce_mean(tf.math.square(tf.stop_gradient(z_e) - z_q))
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(x, p_x_given_z_q))

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
