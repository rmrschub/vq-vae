import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from residual_block import ResidualBlock


tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

@tf.keras.saving.register_keras_serializable()
class CategoricalVectorQuantizedVAE(tf.keras.Model):

    def __init__(self, **kwargs):
        super(CategoricalVectorQuantizedVAE, self).__init__()

        self.__dict__.update(kwargs)

        self.commitment_loss_tracker = tf.keras.metrics.Mean(name="commitment_loss")
        self.codebook_loss_tracker = tf.keras.metrics.Mean(name="codebook_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

        self._prior = tfd.Categorical(
           probs=tf.ones((self.num_embeddings,)) * (1.0 / self.num_embeddings),
            dtype=tf.int32,
            force_probs_to_zero_outside_support=False,
            validate_args=False,
            allow_nan_stats=True,
        )

        self._encoder = tf.keras.Sequential([
            # Encoder
            tfkl.Conv2D(filters=256, kernel_size=4, strides=2, padding='same'),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(alpha=0.2),
            tfkl.Conv2D(filters=256, kernel_size=4, strides=2, padding='same'),
            ResidualBlock(filters=256, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
            ResidualBlock(filters=256, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
            tfkl.Conv2D(filters=self.num_embeddings, kernel_size=1, strides=1, padding='same', activation=None),
            tfpl.DistributionLambda(
                make_distribution_fn=lambda t: tfd.Categorical(
                    logits=t,
                    dtype=tf.int32,
                    force_probs_to_zero_outside_support=False,
                    validate_args=False,
                    allow_nan_stats=True,
                ),
            )],
            name='encoder', 
        )

        self._codebook = tfkl.Embedding(
            self.num_embeddings,
            self.latent_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(
                minval=(-1.0 / self.num_embeddings), 
                maxval=(1.0 / self.num_embeddings),
                seed=self.random_seed,
            ),
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
            sparse=False,
            trainable=True,
            name='codebook'
        )

        self._decoder = tf.keras.Sequential([
            tfkl.Conv2D(filters=256, kernel_size=1, strides=1, padding='same'),
            ResidualBlock(filters=256, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
            ResidualBlock(filters=256, kernel_size=(3, 3), stride=(1, 1), alpha=0.2),
            tfkl.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same'),
            tfkl.BatchNormalization(),
            tfkl.LeakyReLU(alpha=0.2),
            tfkl.Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same'),
            tfkl.Conv2DTranspose(filters=3, kernel_size=4, strides=1, padding='same')],
            name='decoder'
        )

    def call(self, inputs, training=None, mask=None):
        # Categorical over codebook indices with event shape: (batch, 8, 8)
        p_i_given_x = self._encoder(inputs)

        # Quantized latent representation with output shape (batch, 8, 8, latent_dim)
        quantized_codes = self._codebook(p_i_given_x.sample(seed=self.random_seed))

        # Reconstructed images with output shape (batch, 32, 32, 3)
        x_hat = self._decoder(quantized_codes) 

        return p_i_given_x, quantized_codes, x_hat

    def train_step(self, x):
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            p_i_given_x, quantized_codes, x_hat = self(x, training=True)

            reconstruction_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(x, x_hat))
            kl_loss = tf.reduce_mean(self._prior.kl_divergence(p_i_given_x))

            total_loss = sum([
                kl_loss,
                reconstruction_loss,
                # self.commitment_cost_factor * commitment_loss, 
                # self.quantization_loss_factor * codebook_loss, 
            ])

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # self.commitment_loss_tracker.update_state(commitment_loss)
        # self.codebook_loss_tracker.update_state(codebook_loss)

        self.kl_loss_tracker.update_state(kl_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.total_loss_tracker.update_state(total_loss)    

        return {
            "kl_loss": self.kl_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            # "commitment_loss": self.commitment_loss_tracker.result(),
            # "codebook_loss": self.codebook_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result(),
        }


if __name__ == "__main__":

    import tensorflow as tf
    import tensorflow_probability as tfp
    import tensorflow_datasets as tfds
    from cat_vq_vae import CategoricalVectorQuantizedVAE
    

    _, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
    test_ds = test_ds.map(lambda image, label: tf.divide(tf.cast(image, tf.float32), 255.0)).batch(1)  

    model = CategoricalVectorQuantizedVAE(
        input_dims=(32, 32, 3),
        latent_dim=128,
        num_embeddings=64,
        commitment_cost_factor=0.25,
        quantization_loss_factor=0.99,
        alpha=0.2,
        bernstein_order=3,
        random_seed=43,
        margin=10
    )

    model.build((None, 32, 32, 3))
    model.compile(
        optimizer=tf.optimizers.Adam(),

    )
    model.summary()

    model.fit(
        test_ds
    )