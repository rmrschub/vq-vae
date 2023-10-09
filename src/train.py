import os
import shutil
from pathlib import Path
import json
from box import ConfigBox
from ruamel.yaml import YAML

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors

from dvclive import Live
from dvclive.keras import DVCLiveCallback

from vq_vae import VectorQuantizedVAE

yaml = YAML(typ="safe")


def train():
    # Read DVC configuration
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

    # Configure cross device communication 
    implementation = tf.distribute.experimental.CommunicationImplementation.RING
    communication_options = tf.distribute.experimental.CommunicationOptions(implementation=implementation)

    # Define distribution strategy for synchronous training on multiple workers.
    strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)
    global_batch_size = params.train.batch_size_per_replica * strategy.num_replicas_in_sync

    with strategy.scope():
        # Define, build and compile model within strategy scope
        model = VectorQuantizedVAE(
            input_dims=params.model.input_dims,
            latent_dim=params.model.latent_dim,
            num_embeddings=params.model.num_embeddings,
            commitment_cost=params.model.commitment_cost,
            initializer=params.model.initializer,
            alpha=params.model.alpha,
        )
        model.build((None, 32, 32, 3))

        optimizer = tf.optimizers.Adam(
            learning_rate=float(params.train.learning_rate)
        )

        model.compile(
            optimizer=optimizer,
            metrics=['commitment_loss', 'coeebook_loss', 'reconstruction_loss', 'total_loss']
        )

        # Configure distributed training and test pipelines
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        global_batch_size = params.train.batch_size_per_replica * strategy.num_replicas_in_sync

        train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
        train_ds = train_ds.map(lambda image, label: tf.divide(tf.cast(image, tf.float32), 255.0))
        # normalizer = tfkl.Normalization()
        # normalizer.adapt(train_ds)
        # train_ds = train_ds.map(lambda image: normalizer(image))
        train_ds = train_ds.cache()
        train_ds = train_ds.shuffle(10 * global_batch_size)
        train_ds = train_ds.batch(global_batch_size)
        train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
        train_ds = train_ds.with_options(options)

        # test_ds = test_ds.map(lambda image, label: tf.divide(tf.cast(image, tf.float32), 255))
        # test_ds = test_ds.map(lambda image: tfkl.Normalization(mean=0.0, variance=1.0)(image))
        # test_ds.cache()
        # test_ds = test_ds.shuffle(10 * global_batch_size)
        # test_ds = test_ds.batch(global_batch_size)
        # test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
        # test_ds = test_ds.with_options(options)

    # Get current worker's task_type and task_id
    task_type, task_id = (
        strategy.cluster_resolver.task_type,
        strategy.cluster_resolver.task_id,
    )

    if task_id==0:
        with Live() as live:
            model.fit(
                train_ds,
                epochs=params.train.epochs, 
                batch_size=global_batch_size,
                # validation_data=val_ds,
                # validation_freq=params.train.validation_freq,
                # validation_steps=None,
                verbose=params.train.verbosity,
                callbacks=[DVCLiveCallback(live=live), ],
            )
    else:
        # worker node
        model.fit(
            train_ds,
            epochs=params.train.epochs, 
            batch_size=global_batch_size,
            # validation_data=val_ds,
            # validation_freq=params.train.validation_freq,
            # validation_steps=None,
            verbose=params.train.verbosity,
        )

    # Save trained model(s)
    write_model_path = 'model_{}'.format(task_id) if task_id==0 else 'tmp/model_{}'.format(task_id)
    model.save(write_model_path)

if __name__ == "__main__":
    train()
