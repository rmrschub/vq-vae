import tensorflow as tf
import numpy as np

tfkl = tf.keras.layers


@tf.keras.saving.register_keras_serializable()
class VectorQuantizer(tf.keras.layers.Layer):
    
    def __init__(self, **kwargs):
        super(VectorQuantizer, self).__init__(name='VectorQuantizer')

        # region: Set attributes
        self.embedding_dim = kwargs['embedding_dim']
        self.num_embeddings = kwargs['num_embeddings']
        self.commitment_cost = kwargs['commitment_cost']
        self.initializer = kwargs['initializer'] 
        # endregion

    def build(self, input_shape):

        # region: Initialize the embeddings which we will quantize.
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.embedding_dim, self.num_embeddings, ),
            initializer=self.initializer,
            trainable=True
        )

        super(VectorQuantizer, self).build(input_shape)

    def call(self, inputs):
        # Flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(inputs)
        flattened_inputs = tf.reshape(inputs, [-1, self.embedding_dim])
        
        # Calculate L2-normalized distance between the inputs and the codes.
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0, keepdims=True)
            - 2 * tf.matmul(flattened_inputs, self.embeddings)
        )

        # Retrieve encodings
        encoding_indices = tf.argmin(distances, axis=1)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)

        # Quantization
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        return quantized, encoding_indices
