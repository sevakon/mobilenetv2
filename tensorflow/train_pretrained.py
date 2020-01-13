from lib.dataloader import Dataloader
from lib.normalizer import Normalizer

import tensorflow as tf
import numpy as np
import argparse


def get_model(input_shape, n_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(n_classes, activation='softmax')
            ])
    model.summary()
    return model


def get_model_with_nn_head(input_shape, n_classes):
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(.1),
                tf.keras.layers.Dense(n_classes, activation='softmax')
            ])
    model.summary()
    return model


def train(config, fold_idx):
    print(' ... TRAIN MODEL ON {} FOLD'.format(fold_idx))
    loader = Dataloader(img_size=config.input_size,
                        n_folds=config.n_folds, seed=config.seed)
    loader = loader.fit(config.folder)

    classes = loader.classes

    train, train_steps = loader.train(batch_size=config.batch_size,
                         fold_idx=fold_idx, normalize=False)
    val, val_steps = loader.val(64, fold_idx)

    model = get_model((config.input_size, config.input_size, 3), len(classes))

    model.compile(optimizer=tf.keras.optimizers.Adam(),  # Optimizer
                  # Loss function to minimize
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  # List of metrics to monitor
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])


    filepath="model/pretrained_mobilenet_fold{}".format(fold_idx)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                    monitor='val_acc',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='max')
    logdir = "logs/fold{}/".format(fold_idx)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    callbacks = [checkpoint, tensorboard]

    model.fit(train.repeat(),
              epochs=config.epochs,
              steps_per_epoch = train_steps,
              validation_data=val.repeat(),
              validation_steps=val_steps,
              callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required arguments
    parser.add_argument(
        "-f",
        "--folder",
        required=True,
        help="Path to directory containing images")
    # Optional arguments.
    parser.add_argument(
        "-s",
        "--input_size",
        type=int,
        default=224,
        help="Input image size.")
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=2,
        help="Number of images in a training batch.")
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.")
    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=42,
        help="Seed for data reproducing.")
    parser.add_argument(
        "-n",
        "--n_folds",
        type=int,
        default=5,
        help="Number of folds for CV Training")
    args = parser.parse_args()

    for fold_idx in range(args.n_folds):
        train(args, fold_idx)
