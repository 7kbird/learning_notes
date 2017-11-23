# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 16:16:42 2017

@author: doctor
"""

import os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import ZeroPadding2D, Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.utils.data_utils import get_file
from . import base


class Vgg16(object):
    vgg_mean = np.array([123.68, 116.779, 103.939])

    def __init__(self, cache_dir=None):
        self.model, self.classes = self.create(cache_dir=cache_dir)
        self.class_mode = 'categorial'

    @staticmethod
    def get_batches(path_dir, generator=None, shuffle=True, batch_size=8, class_mode='categorical', **kwargs):
        '''Setup preprocessing for vgg16

        example:

        '''
        if not generator:
            generator = ImageDataGenerator()

        # To make best use of cpu, preprocessing is added in batch loading
        def _norm_img(img):
            # input_shape=(3,224,224)  of (depth, height, width)
            # Mean of each channel as provided by VGG researchers
            vgg_mean = Vgg16.vgg_mean.reshape((3, 1, 1))
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

    @staticmethod
    def img_cvt(images):
        vgg_mean = Vgg16.vgg_mean.reshape((3, 1, 1))
        if len(images.shape) == 3:
            # for only 1 image
            return images[::-1] + vgg_mean  # convert bgr to rgb
        else:
            return images[:, ::-1] + vgg_mean  # convert bgr to rgb

    def create(self, cache_dir=None):
        url_base = 'http://files.fast.ai/models/'

        def _conv_block(model, layers, filters, input_shape=None):
            '''ZeroPad+Convolution+MaxPool

            :param layers: number of convolution layers
            :param filters: number of filters in each layer
            '''
            data_format = 'channels_first'
            for _ in range(layers):
                kwargs = {}
                if input_shape:
                    kwargs = {'input_shape': input_shape}
                    input_shape = None
                model.add(ZeroPadding2D((1, 1), data_format=data_format,
                                        **kwargs))  # Zero pad with 1 pix
                model.add(Conv2D(filters, (3, 3), data_format=data_format,
                                 activation='relu'))  # convolution layer with 3*3 kernel and stride (1,1) by default
            model.add(MaxPool2D((2, 2), strides=(2, 2), data_format=data_format))

        def _dense_block(model):
            model.add(Dense(4096, activation='relu'))
            model.add(Dropout(0.5))

        model = keras.models.Sequential()

        # Convolution layers
        _conv_block(model, 2, 64, input_shape=(3, 224, 224))  # Keras need input_shape in first layer
        for layers, filters in [(2, 128), (3, 256), (3, 512), (3, 512)]:
            _conv_block(model, layers, filters)

        model.add(Flatten())
        # Dense layers
        _dense_block(model)
        _dense_block(model)
        model.add(Dense(1000, activation='softmax'))  # the trained ImageNet has 1000 classes as output

        # Download precompiled model weights
        if not cache_dir:
            cache_dir = base.cache_dir('vgg16', 'pre-trained')
        weights = get_file('vgg16.h5', url_base + 'vgg16.h5', cache_subdir=cache_dir,
                           hash_algorithm='md5', file_hash='884146ea83b6c8120d28f686b56eb471')
        model.load_weights(weights)

        # Download label classes
        class_file = get_file('imagenet_class_index.json', url_base + 'imagenet_class_index.json',
                           cache_subdir=cache_dir)
        with open(class_file) as f:
            import json
            class_dict = json.load(f)
        classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

        return model, classes

    def finetune(self, batches, class_mode='categorial', method='replace',
                 **kwargs):
        '''Change the last layer to fit for the batches. Freeze other layer weights

        :param batches: convert the model to fit for the batch data
        :return: None
        '''
        if class_mode not in ['categorial']:
            raise NotImplementedError(class_mode + ' is not implemented yet')
        if method not in ['replace', 'append', 'none']:
            raise NotImplementedError(method + ' is not implemented yet')

        self.class_mode = class_mode

        if method != 'none':
            if method == 'replace':
                self.model.pop()

            for layer in self.model.layers:
                layer.trainable = False

        self.model.add(Dense(batches.num_class, activation='softmax'))


        self.compile(**kwargs)

        # Find classes
        class_dic = batches.class_indices

        # Keras class_indices is in form of {'classA':0, 'classB':1,...}
        self.classes = sorted([name for name in class_dic], key=lambda name: class_dic[name])

    def compile(self, lr=0.001, optimizer=None):
        if not optimizer:
            from keras.optimizers import Adam
            optimizer=Adam(lr=lr)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _wrap_prediction(self, predicts, class_mode):
        if class_mode == 'auto':
            class_mode = self.class_mode

        if class_mode == 'categorial':
            indexes = np.argmax(predicts, axis=1)
            predicts = [predicts[i, idx] for i, idx in enumerate(indexes)]
            classes = [self.classes[i] for i in indexes]

            return predicts, indexes, classes
        elif class_mode is None:
            return predicts

    def predict_data(self, images, class_mode='auto'):
        predicts = self.model.predict(images)

        return self._wrap_prediction(predicts, class_mode)

    def predict(self, batch_generator, class_mode='auto', verbose=0):
        steps = int(np.ceil(batch_generator.samples //
                            batch_generator.batch_size))
        predicts = self.model.predict_generator(batch_generator, workers=1,
                                                steps=steps, verbose=verbose)
        return self._wrap_prediction(predicts, class_mode)

    def fit(self, batches, val_batches, epochs, **kwargs):
        self.model.fit_generator(batches,
                                 steps_per_epoch=batches.samples //
                                                 batches.batch_size,
                                 epochs=epochs,
                                 validation_data=val_batches,
                                 validation_steps=val_batches.samples //
                                                  val_batches.batch_size,
                                 **kwargs)

    def _default_weight_path(self, file_name):
        return os.path.join(base.cache_dir('vgg16', 'weights'), file_name)

    def save_weights(self, file_name):
        if not os.path.isabs(file_name):
            file_name = self._default_weight_path(file_name)
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        self.model.save_weights(file_name)

        return file_name

    def load_weights(self, file_name):
        if not os.path.isabs(file_name):
            file_name = self._default_weight_path(file_name)
        if not os.path.isfile(file_name):
            raise RuntimeError("File not exists:", file_name)

        self.model.load_weights(file_name)
