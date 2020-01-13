from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from lib.normalizer import Normalizer
import tensorflow as tf
import numpy as np
import math
import os


class Dataloader(object):
    """
    A class that manages loading image datasets from folders
    to TensroFlow Dataset types that are ready for training.
    Handles training data normalization, spliiting into train
    and val datasets based on folds, does data augmentation

    Attributes
    ----------
    img_size : int
        Image size to which all of the images will be transformed
    n_folds : int
        Number of folds to divide dataset into
    seed : int
        Number representing random SEED
    size : int
        Number of images inside the dataset
    classes : [ClassInfo]
        List of ClassInfo, each of ClassInfo contains info about
        class in dataset: its name, absolute path, size, index
    normalizer : Normalizer
        Custom class implenting Welford's online algorithm for
        calculating mean and std, then scales and centers data
        respectively to the calculated mean and standard deviation
    filenames : [str]
        List of absolute paths to all images
    labels : numpy.array
        Numpy array of shape (number of images in dataset,
        number of classes in dataset); contains labels for training

    Methods
    -------
    fit(path)
        fitting the loader with path to image data
    train(batch_size, fold, augment)
        Returns training dataset
    val(batch_size, fold)
        Returns validation dataset

    """
    def __init__(self, img_size, n_folds, seed):
        self.img_size = img_size
        self.n_folds = n_folds
        self.seed = seed
        if n_folds > 1:
            self.kf = KFold(n_splits=n_folds, random_state=seed)

    def fit(self, path, png_to_jpg=False):
        """ Fitting the loader with image data
        Args:
            path: path to dataset folder, str
            png_to_jpg: if needed to convert all png
                        images to jpg, bool, optional
        Returns:
            self
        """
        self._analyze_path(path)
        if png_to_jpg:
            self._convert_png_to_jpg(path)
        self._generate_tensor_slices()
        return self

    def train(self, batch_size, fold_idx, normalize=True, augment=False):
        """ Dataset for model training
        Args:
            batch_size: int, number of images in a batch
            fold_idx: int, index of fold, from 0 to n_folds - 1
            normalize: bool, wether to normalize training data
                                    with Welford's online algorithm.
            augment: bool, wether to use augmentation or not
        Returns:
            data: TensroFlow dataset
            steps: int, number of steps in train epoch
        """
        if not (fold_idx >= 0 and fold_idx < self.n_folds):
            raise Exception(('Fold index {} is out of expected range:'
                    + '  [0, {}]').format(fold_idx, self.n_folds - 1))

        if normalize and augment:
            raise Exception('Both augmentations and normalization ' +
                                'with Welford algo is not supported ')

        print(' ... Generating Training Dataset ... ')
        if self.n_folds == 1:
            train_idx = range(0, len(self.filenames))
        else:
            train_idx, _ = list(self.kf.split(self.filenames))[fold_idx]
        filenames = np.array(self.filenames)[train_idx]
        labels = np.array(self.labels)[train_idx]
        steps = math.ceil(len(filenames)/batch_size)
        if normalize:
            mean, std = Normalizer.calc_mean_and_std(filenames, self.img_size)
            mean = np.array((mean['red'], mean['green'], mean['blue']))
            std = np.array((std['red'], std['green'], std['blue']))
        else:
            # values taken from ImageNet Dataset
            mean = np.array(0.485, 0.456, 0.406)
            std = np.array(0.229, 0.224, 0.225)
            self.normalizer = Normalizer(mean, std)
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(filenames), tf.constant(labels))
        )
        data = data.map(self.parse_fn)
        if augment:
            augs = [self.flip, self.color, self.rotate, self.zoom]
            for f in augs:
                data = data.map(f, num_parallel_calls=4)
            data = data.map(self.drop, num_parallel_calls=4)
        data = data.shuffle(buffer_size=len(filenames))
        data = data.batch(batch_size)
        data = data.prefetch(1)
        return data, steps

    def val(self, batch_size, fold_idx):
        """ Dataset for data validation
        Args:
            batch_size: int, number of images in a batch
            fold_idx: int, index of fold, from 0 to n_folds - 1
        Returns:
            data : TensroFlow dataset
            steps : number of steps in val epoch
        """
        if not (fold_idx >= 0 and fold_idx < self.n_folds):
            raise Exception(('Fold index {} is out of expected range:'
                    + '  [0, {}]').format(fold_idx, self.n_folds - 1))
        _, test_idx = list(self.kf.split(self.filenames))[fold_idx]
        filenames = np.array(self.filenames)[test_idx]
        labels = np.array(self.labels)[test_idx]
        steps = math.ceil(len(filenames)/batch_size)
        data = tf.data.Dataset.from_tensor_slices(
            (tf.constant(filenames), tf.constant(labels))
        )
        data = data.map(self.parse_fn).batch(batch_size)
        data = data.prefetch(1)
        return data, steps


    def parse_fn(self, filename, label):
        ''' Parsing filename, label pair into float array '''
        ''' Performing normalization, size transformation '''
        img = tf.io.read_file(filename)
        img = tf.image.decode_jpeg(img)
        img = tf.image.resize(img, (self.img_size, self.img_size))
        img = tf.cast(img, tf.float32) / 255
        img = self.normalizer(img)
        return img, label

    def _analyze_path(self, path):
        ''' Analyzing defined path, extracting class names '''
        print(' ... Checking "{}" path '.format(path))

        if not os.path.exists(path):
            raise Exception('Specified path does not exist')

        self.classes = [ClassInfo(f.name, f.path) for f in \
                                os.scandir(path) if f.is_dir() \
                                            and f.name != 'other']
        if len(self.classes) == 0:
            raise Exception('Specified path has no folders')

        print(' ... Found {} classes,  \
                \n ... listing them below:'.format(len(self.classes)))
        self.size = 0
        for item in self.classes:
            self.size += item.size
            print(' ...   [{}] {} images '.format(item.name, item.size))
        print(' ... Total image number: {}'.format(self.size))

    def _generate_tensor_slices(self):
        ''' Generates filenames and labels for model training '''
        print(' ... Generating {} filenames & labels'.format(self.size))
        idx, counter = 0, 0
        self.filenames = []
        self.labels = np.zeros((self.size, len(self.classes)),
                                                    dtype='float32')
        for item in self.classes:
            item.index = counter
            paths = [os.path.join(item.path, f) for f \
                        in os.listdir(item.path) if f.endswith('jpg')]
            self.filenames += paths
            self.labels[idx:idx + item.size, item.index] = 1.0
            idx += item.size
            counter += 1
        self.filenames, self.labels = shuffle(self.filenames, \
                                self.labels, random_state=self.seed)

    def _convert_png_to_jpg(self, path):
        ''' Converts all png images in dataset to jpg images '''
        for item in self.classes:
            png_images = [f for f in os.listdir(item.path) \
                                                 if f.endswith('png')]
            print(' ... Converting {} {} PNG images to JPG ...'.\
                                  format(len(png_images), item.name))
            for img in png_images:
                self.png_to_jpg(img)

    @staticmethod
    def png_to_jpg(path):
        ''' Converts image from PNG to JPG
        Args:
            path: path to image, str
        '''
        im = Image.open(path)
        rgb_im = im.convert('RGB')
        new_path = path[:-3] + 'jpg'
        rgb_im.save(new_path)
        os.remove(path)

    # ----------------------- AUGMENTATION ------------------------ #
    @staticmethod
    def flip(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """Flip augmentation
        Args:
            x: Image to flip
            y: its label
        Returns:
            (Augmented image, label)
        """
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_flip_up_down(x)

        return x, y

    @staticmethod
    def color(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """Color augmentation
        Args:
            x: Image
            y: label
        Returns:
            (Augmented image, label)
        """
        x = tf.image.random_hue(x, 0.08)
        x = tf.image.random_saturation(x, 0.6, 1.6)
        x = tf.image.random_brightness(x, 0.05)
        x = tf.image.random_contrast(x, 0.7, 1.3)

        return x, y

    @staticmethod
    def rotate(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """Rotation augmentation
        Args:
            x: Image
            y: label
        Returns:
            (Augmented image, label)
        """

        return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0,
                                        maxval=4, dtype=tf.int32)), y

    def zoom(self, x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        """Zoom augmentation
        Args:
            x: Image
            y: label
        Returns:
            (Augmented image, label)
        """
        # Generate 20 crop settings, ranging from a 1% to 20% crop.
        scales = list(np.arange(0.8, 1.0, 0.01))
        boxes = np.zeros((len(scales), 4))

        for i, scale in enumerate(scales):
            x1 = y1 = 0.5 - (0.5 * scale)
            x2 = y2 = 0.5 + (0.5 * scale)
            boxes[i] = [x1, y1, x2, y2]

        def random_crop(img):
            # Create different crops for an image
            crops = tf.image.crop_and_resize([img], boxes=boxes,
                        box_indices=np.zeros(len(scales)),
                            crop_size=(self.img_size, self.img_size))
            # Return a random crop
            return crops[tf.random.uniform(shape=[], minval=0,
                                maxval=len(scales), dtype=tf.int32)]

        choice = tf.random.uniform(shape=[], minval=0.,
                                        maxval=1., dtype=tf.float32)
        # Only apply cropping 50% of the time
        return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x)), y

    @staticmethod
    def drop(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        x = tf.clip_by_value(x, 0, 1)
        return x, y



class ClassInfo(object):
    """
    A class that stores information about class inside the dataset

    Attributes
    ----------
        name : str
            a string containing class name
        path: str
            a string contating absolute path to the class folder
        size: int
            a number of image belonging to this class
        index: int
            an integer representing class number in the dataset,
            lies within [0; n_classes]; this number is the output of
                                                    the trained model
    """
    def __init__(self, name, path):
        self.name = name
        self.path = path
        images = [f for f in os.listdir(path) if f.endswith('jpg') or \
                               f.endswith('jpeg') or f.endswith('png')]
        self.size = len(images)
        self.index = None # create a dummy for further initialization
