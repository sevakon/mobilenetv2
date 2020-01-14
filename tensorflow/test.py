from dataloader import Dataloader
from normalizer import Normalizer

import tensorflow as tf
import numpy as np
import argparse

def load(config):
    loader = Dataloader(img_size=config.input_size,
                        n_folds=config.n_folds, seed=config.seed)
    loader = loader.fit(config.folder)


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
    load(args)
