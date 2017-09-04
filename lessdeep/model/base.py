import os
import keras

__default_cache_root=os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '_model'))


def cache_dir(name, typename='cache'):
    return os.path.join(__default_cache_root, name, typename)


def get_batches(path_dir, generator, shuffle=True, batch_size=8, class_mode='categorial', **kwargs):
    '''Load batches from directory and return a batch generator
    '''
    if type(generator) is keras.preprocessing.image.ImageDataGenerator:
        batches = generator.flow_from_directory(directory=path_dir, shuffle=shuffle,
                                                batch_size=32, **kwargs)
        if batches.samples < batch_size:
            batches.batch_size = batches.samples  # keras raise error when batch_size is larger than total samples
        return batches
    else:
        raise NotImplemented('Generator %s is not supported' % generator)