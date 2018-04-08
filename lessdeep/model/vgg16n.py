# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from . import base
from keras.layers import Dense, Dropout, BatchNormalization
from functools import partial
from ..utils import load_array, clone_model


class Vgg16N(object):
    """VGG16 with normalization between dense layers"""

    # Mean of each channel(RGB) as provided by VGG researchers
    vgg_mean = np.array([123.68, 116.779, 103.939])

    def __init__(self, include_top=True, dropout=None, batch_norm=False,
                 image_size=None, pooling=None, **kwargs):
        """

        :param include_top:
        :param dropout: Can be `None`, float, tuple(drop1, drop2)
        :param batch_norm: bool, use batch normalization between dense
        :param kwargs:
            classes: list of class that to be predicted
        """
        if batch_norm and dropout is not None:
            raise ValueError('If use `batchnorm` as true, `dropout` should be '
                             'None')
        if image_size:
            if K.image_data_format() == 'channels_first':
                input_shape = (3, ) + tuple(image_size)
            else:
                input_shape = tuple(image_size) + (3, )
        else:
            input_shape = None
        model = keras.applications.VGG16(include_top, input_shape=input_shape,
                                         pooling=pooling, **kwargs)
        if batch_norm:
            raise NotImplementedError('Batch normalization is not '
                                      'implemented yet')

        if include_top and dropout is not None:
            if isinstance(dropout, (tuple, list)):
                if len(dropout) != 2:
                    raise ValueError('Dropout must be tuple of two float')
            elif isinstance(dropout, float):
                dropout = (dropout, dropout)
            else:
                raise ValueError('`dropout` must be `None`, `float` or tuple '
                                 'of two floats')

            dense_layers = [l for l in model.layers if type(l) == Dense]
            for dense_layer, drop_rate in zip(dense_layers[:-1], dropout):
                # add dropout between each dense layers
                idx = model.layers.index(dense_layer)
                weights = dense_layer.get_weights()
                # rescale weight according to dropout rate
                dense_layer.set_weights([w / (1.0 - drop_rate) for w in weights])
                dropout_layer = Dropout(drop_rate)
                model.layers.insert(idx + 1, dropout_layer)
        if not include_top:
            # Remove last MaxPooling from Convolution Layers to prevent
            # down sampling
            model.layers.pop()

        self.model = clone_model(model.layers)
        self.classes = kwargs.get('classes', [])

    def get_batches(self, path_dir, batch_size, generator=None, shuffle=True,
                    class_mode='categorical', image_size=None, **kwargs):
        """Setup pre-processing for vgg16 and return batch generator

        :param self:
        :param path_dir:
        :param batch_size:
        :param generator: ImageDataGenerator like generator
        :param shuffle:
        :param class_mode:
        :param kwargs:
        :return:
        """
        if not generator:
            generator = ImageDataGenerator()

        # To make best use of cpu, pre-processing is added on batch loading
        from keras.applications.imagenet_utils import preprocess_input
        if generator.preprocessing_function:
            def wrapper(x, f1, f2):
                return f1(f2(x))
            generator.preprocessing_function = \
                partial(wrapper, preprocess_input,
                        generator.preprocessing_function)
        else:
            generator.preprocessing_function = preprocess_input

        if image_size and len(image_size) != 2:
            raise ValueError('`expect `image_size` be a tuple of size 2')
        if not image_size:
            image_size = (224, 224)     # ImageNet image size is 244x244

        return base.get_batches(path_dir, generator, shuffle=shuffle,
                                batch_size=batch_size, class_mode=class_mode,
                                target_size=image_size, **kwargs)

    def compile(self, lr=0.001, optimizer=None):
        if not optimizer:
            from keras.optimizers import Adam
            optimizer=Adam(lr=lr)
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def _wrap_prediction(self, predicts, class_mode, top=1):
        if class_mode == 'categorial':
            if self.classes:
                if len(predicts.shape) != 2 or \
                        predicts.shape[1] != len(self.classes):
                    raise ValueError('`prediction` expects '
                                     'a batch of predictions '
                                     '(i.e. a 2D array of shape (samples, '+
                                     str(len(self.classes)) + ')). Found array '
                                     'with shape: ' + str(predicts.shape))
                results = []
                for pred in predicts:
                    top_indices = pred.argsort()[-top:][::-1]
                    result = [(i, self.classes[i], pred[i]) for i in
                              top_indices]
                    if top == 1:
                        result = result[0]
                    else:
                        result.sort(key=lambda x: x[2], reverse=True)
                    results.append(result)
                return results
            else:
                raise ValueError('Vgg16N.classes is not set')
        elif class_mode is None:
            return predicts

    def predict_data(self, images, class_mode='categorial', top=1):
        predicts = self.model.predict(images)

        return self._wrap_prediction(predicts, class_mode, top)

    def predict(self, batch_generator, class_mode='categorial',
                top=1, verbose=1):
        self.classes = sorted(batch_generator.class_indices.items(),
                              key=lambda x: x[1])
        self.classes = [x[0] for x in self.classes]

        steps = int(np.ceil(batch_generator.samples //
                            batch_generator.batch_size))
        predicts = self.model.predict_generator(batch_generator, workers=1,
                                                steps=steps, verbose=verbose)

        return self._wrap_prediction(predicts, class_mode, top)

    def fit(self, batches, val_batches, epochs, **kwargs):
        self.model.fit_generator(batches, steps_per_epoch=batches.samples //
                                                          batches.batch_size,
                                 epochs=epochs, validation_data=val_batches,
                                 validation_steps=val_batches.samples //
                                                  val_batches.batch_size,
                                 **kwargs)

    def _default_weight_path(self, file_name):
        return os.path.join(base.cache_dir('vgg16n', 'weights'), file_name)

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

    @staticmethod
    def _default_predict_root(file_root):
        if not Path(file_root).is_absolute():
            file_root = str(Path(base.cache_dir('vgg16n', 'predict'),
                                 file_root))
        return str(Path(file_root, 'x')), str(Path(file_root, 'y'))

    def save_predict(self, batches, file_root, samples=0, save_label=True):
        # TODO: save classes
        import bcolz
        import progressbar

        if not samples:
            # By default: use all samples only once
            samples = batches.samples
        bar = progressbar.ProgressBar()
        if hasattr(self.model.output.dtype, 'itemsize'):
            # For numpy dtype
            itemsize = self.model.output.dtype.itemsize
        elif hasattr(self.model.output.dtype, 'size'):
            # For tensorflow dtype
            itemsize = self.model.output.dtype.size
        else:
            raise ValueError("Model cannot determine output dtype size:" + \
                             str(self.model.output.dtype))
        outshape = self.model.output.shape.as_list()[1:]
        if None in outshape:
            tmp_x, _ = batches.next()
            model = keras.Sequential([keras.layers.InputLayer(tmp_x.shape[1:])] + \
                                     self.model.layers[1:])
            outshape = list(model.output_shape)[1:]
        batch_bytes = np.prod(outshape) * \
            batches.batch_size * itemsize
        # use 512M per step to prevent little garbage collect
        max_step_bytes = 512 * 1024 * 1024
        step = int(np.ceil(max_step_bytes / batch_bytes))
        batch_size = batches.batch_size
        total_steps = int(np.ceil(samples / (step * batch_size)))

        # create a generator which save y when generating

        def _gen(start, y_array):
            def _save_return_next():
                x, y = batches.next()
                y_array.append(y)
                return x, y
            batches.batch_index = start
            if save_label:
                while True:
                    yield _save_return_next()
            else:
                while True:
                    yield batches.next()

        path_x, path_y = Vgg16N._default_predict_root(file_root)
        os.makedirs(path_x, exist_ok=True)
        if save_label:
            os.makedirs(path_y, exist_ok=True)
        # Start progress
        for i in bar(range(total_steps)):
            y_array = []
            predicted_tmp = self.model.predict_generator(
                                _gen(i * step, y_array), steps=step, workers=1)
            if save_label:
                y_label = np.vstack(y_array[:step])
            if i == 0:
                # Lazy initialization
                if total_steps == 1:
                    predicted_tmp = predicted_tmp[:samples]
                    y_label = y_label[:samples]
                arr = bcolz.carray(predicted_tmp, mode='w', rootdir=path_x)
                if save_label:
                    arr_y = bcolz.carray(y_label, mode='w', rootdir=path_y)
            elif i != total_steps - 1:
                arr.append(predicted_tmp)
                if save_label:
                    arr_y.append(y_label)
            else:
                last_num = samples - (total_steps - 1) * batch_size * step
                arr.append(predicted_tmp[:last_num])
                if save_label:
                    arr_y.append(y_label[:last_num])
        arr.flush()
        if save_label:
            arr_y.flush()

    @staticmethod
    def get_predict_batch(file_root):
        path_x, path_y = Vgg16N._default_predict_root(file_root)

        # TODO: generate batch generator for large array
        features = load_array(path_x)
        if os.path.exists(path_y):
            labels = load_array(path_y)
            return features, labels
        else:
            return features

    @property
    def last_conv_index(self):
        return [i for i, layer in enumerate(self.model.layers) if type(layer)
                == keras.layers.Conv2D][-1]

    def get_top_layers(self, dense_num=3, clone=False, trainable=False):
        '''Get only top layers

        :param clone: True if clone all the top layers
        :param trainable: set all layers `trainable`, only work when clone
        is True
        :return: the top layers
        '''
        flat_dense_layers = self.model.layers[self.last_conv_index + 1:]
        last_dense_idx = [i for i, layer in enumerate(flat_dense_layers) if \
                          type(layer) == keras.layers.Dense][dense_num - 1]
        flat_dense_layers = flat_dense_layers[:last_dense_idx + 1]

        if clone:
            return clone_model(flat_dense_layers, trainable=trainable)
        return flat_dense_layers
