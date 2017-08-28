# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:16:42 2017

@author: doctor
"""

import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from . import base


class Vgg16(object):
    def __init__(self):
        pass

    @staticmethod
    def get_batches(self, path_dir, generator=None, shuffle=True, batch_size=8, class_mode='categorial', **kwargs):
        '''Setup preprocessing for vgg16

        example:

        '''
        if not generator:
            generator = ImageDataGenerator()

        # To make best use of cpu, preprocessing is added in batch loading
        def _norm_img(img):
            # input_shape=(3,224,224)  of (depth, height, width)
            # Mean of each channel as provided by VGG researchers
            vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))
            return (img - vgg_mean)[::-1]  # convert rgb to bgr

        # make sure the data is loaded as (depth, height, width)
        generator.data_format = 'channels_first'

        # add normalization as preprocessing
        if generator.preprocessing_function:
            f = generator.preprocessing_function
            generator.preprocessing_function = lambda x, f_norm=_norm_img, f_old=f: f_norm(f_old(x))
        else:
            generator.preprocessing_function = _norm_img

        # NOTE: target_size must be (224, 224) for the pretrained model so that all image are resized
        return base.get_batches(path_dir, generator, shuffle=shuffle, batch_size=batch_size, class_mode=class_mode,
                                target_size=(224, 224), **kwargs)

