import tensorflow as tf
import numpy as np

tfkl = tf.keras.layers


@tf.keras.saving.register_keras_serializable()
class VectorQuantizer(tf.keras.Model):
    
    def __init__(self, **kwargs):

        super(VectorQuantizer, self).__init__(name='VectorQuantizer')

        # region: Set attributes
        self.latent_dim = kwargs['latent_dim']
        self.num_embeddings = kwargs['num_embeddings']
        # endregion

        self._codebook = tfkl.Embedding(
            self.num_embeddings,
            self.latent_dim,
            embeddings_initializer=tf.keras.initializers.RandomUniform(
                minval=(-1.0 / self.num_embeddings), 
                maxval=(1.0 / self.num_embeddings),
                seed=42,
            ),
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
            sparse=False,
            trainable=True
        )

    def build(self, input_shape):

        self._codebook.build((None, ))

    def call(self, inputs):

        # Flatten the inputs keeping `latent_dim` intact.
        input_shape = tf.shape(inputs)
        flattened_inputs = tf.reshape(inputs, [-1, self.latent_dim])
        
        # Calculate L2-normalized distance between the inputs and the codes.
        distances = tf.reduce_sum(
            (tf.expand_dims(flattened_inputs, 1)-tf.expand_dims(self._codebook.weights[0], 0))**2, 
            2
        )  

        # Retrieve codebook indices and quantized vectors
        codebook_indices = tf.argmin(distances, axis=1)
        quantized = self._codebook(codebook_indices)
        
        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        return codebook_indices, quantized
