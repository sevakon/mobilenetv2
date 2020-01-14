from dataloader import Dataloader
from normalizer import Normalizer
from mobilenetv2 import MobileNetV2

import tensorflow as tf
import numpy as np
import argparse


def write_stats_to_file(mean, std, fold_idx):
    file = open("model/norm_fold{}.txt".format(fold_idx), "x")
    file.write('Mean: r {}, g {}, b {}\n'.format(mean[0], mean[1], mean[2]))
    file.write('Std: r {}, g {}, b {}\n'.format(std[0], std[1], std[2]))
    file.close()


def train(config, fold_idx):
    print(' ... TRAIN MODEL ON {} FOLD'.format(fold_idx))
    loader = Dataloader(img_size=config.input_size,
                        n_folds=config.n_folds, seed=config.seed)
    loader = loader.fit(config.folder)

    classes = loader.classes

    train, train_steps = loader.train(batch_size=config.batch_size,
                         fold_idx=fold_idx, normalize=True)
    val, val_steps = loader.val(1, fold_idx)

    write_stats_to_file(loader.normalizer.mean, stloader.normalizer.stdd, fold_idx)

    model = MobileNetV2((config.input_size, config.input_size, 3), len(classes))

    model.compile(optimizer=tf.keras.optimizers.Adam(),  # Optimizer
                  # Loss function to minimize
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  # List of metrics to monitor
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])


    filepath="model/mobilenet_fold{}.hdf5".format(fold_idx)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath,
                                                    monitor='val_categorical_accuracy',
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
