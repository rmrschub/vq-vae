import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from residual_block import ResidualBlock
from vector_quantizer import VectorQuantizer

tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class VectorQuantizedVAE(tf.keras.Model):

    def __init__(self, **kwargs):
        super(VectorQuantizedVAE, self).__init__()

        # region: Set attributes
        self.input_dims = kwargs['input_dims']
        self.latent_dim = kwargs['latent_dim']
        self.num_embeddings = kwargs['num_embeddings']
        self.commitment_cost = kwargs['commitment_cost']
        self.initializer = kwargs['initializer']
        self.alpha = kwargs['alpha']
        # endregion

        # region Define loss trackers
        self.commitment_loss_tracker = tf.keras.metrics.Mean(name="commitment_loss")
        self.codebook_loss_tracker = tf.keras.metrics.Mean(name="codebook_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # endregion

        # bx32x32x3
        self._input_layer = tfkl.Input(shape=self.input_dims, name='input_layer')

        self._encoder = tf.keras.Sequential([
                tfkl.Conv2D(filters=256, kernel_size=4, strides=2, padding='same'),         # bx16x16x256
                tfkl.BatchNormalization(),
                tfkl.LeakyReLU(alpha=0.2),
                tfkl.Conv2D(filters=self.latent_dim, kernel_size=4, strides=1, padding='same'),         # bx16x16x256
                ResidualBlock(filters=self.latent_dim, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
                ResidualBlock(filters=self.latent_dim, kernel_size=(3, 3), stride=(1, 1), alpha=0.2)],  # bx16x16xD
            name='encoder', 
        )

        self._vector_quantizer = VectorQuantizer(           
            num_embeddings=self.num_embeddings,
            embedding_dim=self.latent_dim,
            commitment_cost=self.commitment_cost, 
            initializer='uniform'
        )

        self._straight_through_estimator = tfkl.Lambda(
            lambda z_e_q: z_e_q[0] + tf.stop_gradient(z_e_q[1] - z_e_q[0]), 
            name='straight_through_estimator'
        )

        self._decoder = tf.keras.Sequential([
                ResidualBlock(filters=self.latent_dim, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),       # 8x8xD
                ResidualBlock(filters=self.latent_dim, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
                tfkl.Conv2DTranspose(filters=self.latent_dim, kernel_size=4, strides=1, padding='same'),    # 16x16x256
                tfkl.BatchNormalization(),
                tfkl.LeakyReLU(alpha=0.2),
                tfkl.Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', activation=None),      # 32x32
                tfkl.Flatten(),
                tfpl.IndependentBernoulli((32, 32, 3), tfd.Bernoulli.logits)],
            name='decoder', 
        )

        ## Re-Initialize Model with input_layer inside call method to setup all shape
        input_shape = self._input_layer
        output_shape = self.call(input_shape)
        super(VectorQuantizedVAE, self).__init__(name="VectorQuantizedVAE", inputs=input_shape, outputs=output_shape)

    def call(self, x, training=None, mask=None):
        z_e = self._encoder(x)
        z_q, encoding_indices, commitment_loss, codebook_loss = self._vector_quantizer(z_e)
        p_x_given_z_q = self._decoder(z_q)

        return z_e, z_q, encoding_indices, commitment_loss, codebook_loss, p_x_given_z_q

    def train_step(self, x):
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            z_e, z_q, encoding_indices, commitment_loss, codebook_loss, p_x_given_z_q = self(x, training=True)
           
            reconstruction_loss = -1.0 * p_x_given_z_q.log_prob(x) 

            total_loss = sum([commitment_loss, codebook_loss, reconstruction_loss])
       
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
