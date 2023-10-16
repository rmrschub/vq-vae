import numpy as np
import random
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_datasets as tfds


from vq_vae import VectorQuantizedVAE

tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers


class TripletVectorQuantizedVAE(VectorQuantizedVAE):
    
    def __init__(self, **kwargs):
        super(TripletVectorQuantizedVAE, self).__init__(**dict(kwargs, name="TripletVectorQuantizedVAE"))
        
        # region: Set attributes
        #
        # input_dims
        # latent_dim
        # num_embeddings
        # commitment_cost_factor
        # quantization_loss_factor
        # bernstein_order
        # alpha
        # random_seed
        # margin

        self.__dict__.update(kwargs)

        # endregion

        # region: Define loss trackers

        self.triplet_loss_tracker = tf.keras.metrics.Mean(name="triplet_loss")
        self.triplet_ham_loss_tracker = tf.keras.metrics.Mean(name="triplet_ham_loss")
        
        # endregion
        
    def train_step(self, data):

        imgs, labels = data

        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            
            # get all embeddings
            z_e, codebook_indices, z_q, p_x_given_z_q = self(imgs, training=True)

            # region: Compute triplet loss on codebook indices
            ham_distances = self.pairwise_ham_distance(codebook_indices)
            anchor_positive_ham_dist = tf.expand_dims(ham_distances, 2)
            anchor_negative_ham_dist = tf.expand_dims(ham_distances, 1)
            triplet_ham_loss = tf.cast(anchor_positive_ham_dist - anchor_negative_ham_dist + self.ham_margin, dtype=tf.float32)
            mask = self.get_triplet_mask(labels)
            mask = tf.cast(mask, dtype=tf.float32)
            triplet_ham_loss = tf.cast(tf.multiply(mask, triplet_ham_loss), dtype=tf.float32)
            triplet_ham_loss = tf.maximum(triplet_ham_loss, 0.0)
            valid_triplets = tf.cast(tf.greater(triplet_ham_loss, 1e-16), dtype=tf.float32)
            num_positive_triplets = tf.reduce_sum(valid_triplets)
            num_valid_triplets = tf.reduce_sum(mask)
            fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
            triplet_ham_loss = tf.reduce_sum(triplet_ham_loss) / (num_positive_triplets + 1e-16)

            # endregion

            # region: Compute triplet loss of latent representations
            # compute pairwise distance matrix of latent representations
            distances = self.pairwise_distances(z_q)

            anchor_positive_dist = tf.expand_dims(distances, 2)
            anchor_negative_dist = tf.expand_dims(distances, 1)

             # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
            # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
            # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
            # and the 2nd (batch_size, 1, batch_size)
            triplet_loss = tf.cast(anchor_positive_dist - anchor_negative_dist + self.margin, dtype=tf.float32)

             # Put to zero the invalid triplets
            # (where label(a) != label(p) or label(n) == label(a) or a == p)
            mask = self.get_triplet_mask(labels)
            mask = tf.cast(mask, dtype=tf.float32)
            triplet_loss = tf.cast(tf.multiply(mask, triplet_loss), dtype=tf.float32)

            # Remove negative losses (i.e. the easy triplets)
            triplet_loss = tf.maximum(triplet_loss, 0.0)

            # Count number of positive triplets (where triplet_loss > 0)
            valid_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), dtype=tf.float32)
            num_positive_triplets = tf.reduce_sum(valid_triplets)
            num_valid_triplets = tf.reduce_sum(mask)
            fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

            # Get final mean triplet loss over the positive valid triplets
            triplet_loss = tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)
            # endregion

            commitment_loss = self.commitment_cost_factor * (tf.reduce_mean(tf.math.square(z_e - tf.stop_gradient(z_q))))
            codebook_loss = self.quantization_loss_factor * (tf.reduce_mean(tf.math.square(tf.stop_gradient(z_e) - z_q)))
            reconstruction_loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)(imgs, p_x_given_z_q))

            total_loss = sum([commitment_loss, codebook_loss, reconstruction_loss, triplet_loss, triplet_ham_loss])

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.commitment_loss_tracker.update_state(commitment_loss)
        self.codebook_loss_tracker.update_state(codebook_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.triplet_loss_tracker.update_state(triplet_loss)
        self.triplet_ham_loss_tracker.update_state(triplet_ham_loss)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "triplet_loss": self.triplet_loss_tracker.result(),
            "triplet_ham_loss": self.triplet_ham_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "commitment_loss": self.commitment_loss_tracker.result(),
            "codebook_loss": self.codebook_loss_tracker.result()
        }

    def pairwise_ham_distance(self, inputs):

        distances = tf.math.divide_no_nan(
            tf.reduce_sum(
                tf.cast(
                    tf.not_equal(
                        tf.expand_dims(tf.reshape(inputs, (-1, tf.math.reduce_prod(tf.shape(inputs)[1:]))), 1),
                        tf.expand_dims(tf.reshape(inputs, (-1, tf.math.reduce_prod(tf.shape(inputs)[1:]))), 0)
                    ),
                    tf.float32
                ),
            2),  
            len(a)
        )

        return distances


    def pairwise_distances(self, inputs):
      
        distances = tf.reduce_sum(
            tf.math.square(
                tf.subtract(
                    tf.expand_dims(tf.reshape(inputs, (-1, tf.math.reduce_prod(tf.shape(inputs)[1:]))), 1),
                    tf.expand_dims(tf.reshape(inputs, (-1, tf.math.reduce_prod(tf.shape(inputs)[1:]))), 0)
                )
            ),
            2
        )

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances = tf.maximum(distances, 0.0)

        return distances

    def get_triplet_mask(self, labels):
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.

        A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]

        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """

        # Check that i, j and k are distinct
        indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
        indices_not_equal = tf.logical_not(indices_equal)
        i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
        i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
        j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

        distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)


        # Check if labels[i] == labels[j] and labels[i] != labels[k]
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        i_equal_j = tf.expand_dims(label_equal, 2)
        i_equal_k = tf.expand_dims(label_equal, 1)

        valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))

        # Combine the two masks
        mask = tf.logical_and(distinct_indices, valid_labels)

        return mask


if __name__ == "__main__":
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from triplet_vq_vae import TripletVectorQuantizedVAE

    train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
    train_ds = train_ds.map(lambda image, label: (tf.divide(tf.cast(image, tf.float32), 255.0), label)).cache().shuffle(100).batch(64)

    tvq = TripletVectorQuantizedVAE(
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
    tvq.build((None, 32, 32, 3))

    tvq.compile(
        optimizer='adam',
        metrics=['commitment_loss', 'coeebook_loss', 'reconstruction_loss', 'total_loss', 'triplet_loss']
    )

    tvq.fit(
       train_ds,
        epochs=1,
        verbose=2
    )