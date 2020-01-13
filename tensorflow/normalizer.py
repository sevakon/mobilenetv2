import tensorflow as tf
import numpy as np
import os


class Normalizer(object):
    """
    A class that handles training data normalization
    """
    def __init__(self, mean=None, std=None):
        print('... Initializing normalizer with: ...')
        print('Mean values: {}'.format(mean))
        print('Std values: {}'.format(std))
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if self.mean is not None:
            img = self.__center(img)
        if self.std is not None:
            img = self.__scale(img)
        return img

    def __center(self, img):
        return img - self.mean

    def __scale(self, img):
        return img / self.std

    @staticmethod
    def calc_mean_and_std(filenames, img_size):
        ''' Calculating mean and standard deviation on filenames '''
        '''           with Welford's online algorithm            '''
        print('Calulcating mean and standard deviation on {} train images'\
                                    .format(len(filenames)))
        n_pixels = 0
        mean = {'red': .0, 'green': .0, 'blue': .0}
        M2 = {'red': .0, 'green': .0, 'blue': .0}

        for idx, filename in enumerate(filenames):
            if idx % 100 == 0:
                print('{}/{}'.format(idx + 1, len(filenames)))

            image = tf.io.read_file(filename)
            image = tf.image.decode_jpeg(image)
            image = tf.image.resize(image, (img_size, img_size))
            img = tf.cast(img, tf.float32) / 255
            image = image.numpy()

            red = image[:, :, 0].flatten().tolist()
            green = image[:, :, 1].flatten().tolist()
            blue = image[:, :, 2].flatten().tolist()

            for (r, g, b) in zip(red, green, blue):
                n_pixels += 1
                delta = {'red': r - mean['red'],
                        'green': g - mean['green'],
                        'blue': b - mean['blue']}
                mean['red'] += delta['red']/n_pixels
                mean['green'] += delta['green']/n_pixels
                mean['blue'] += delta['blue']/n_pixels
                M2['red'] += delta['red'] * (r - mean['red'])
                M2['green'] += delta['green'] * (g - mean['green'])
                M2['blue'] += delta['blue'] * (b - mean['blue'])

        variance = {
            'red': M2['red'] / (n_pixels - 1),
            'green': M2['green'] / (n_pixels - 1),
            'blue': M2['blue'] / (n_pixels - 1)
        }

        std = {
            'red': np.sqrt(variance['red']),
            'green': np.sqrt(variance['green']),
            'blue': np.sqrt(variance['blue'])
        }

        for color in ['red', 'green', 'blue']:
            print('{} mean: {}'.format(color, mean[color]))
            print('{} std: {}'.format(color, std[color]))

        return mean, std
