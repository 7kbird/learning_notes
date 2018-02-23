import bcolz
import os
__default_cache_root=os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                   '..', '..', '_cache'))


def cache_dir(name):
    return os.path.join(__default_cache_root, name)


def save_array(arr, file_path):
    if not os.path.isabs(file_path):
        file_path = os.path.join(cache_dir('array'), file_path)
        os.makedirs(file_path, exist_ok=True)
    # TODO: compress level?
    c = bcolz.carray(arr, rootdir=file_path, mode='w')
    c.flush()


def load_array(file_path):
    if not os.path.isabs(file_path):
        file_path = os.path.join(cache_dir('array'), file_path)
    # Open and uncompress
    return bcolz.open(file_path)[:]


def tf_board(prefix='', **kwargs):
    from keras.callbacks import TensorBoard
    from datetime import datetime
    if prefix:
        prefix += '_'
    prefix += datetime.now().strftime("%Y%m%d%H%M%S")

    log_root = cache_dir('tensorboard_logs')
    log_dir = os.path.join(log_root, prefix)
    rename_idx = 0
    while os.path.exists(log_dir):
        log_dir = os.path.join(log_root, prefix + '_' + str(rename_idx))
        rename_idx += 1

    options = {
        'log_dir': log_dir,
        # 'histogram_freq': 0,
        # 'batch_size': 32,
        # 'write_graph': True,
        # 'write_grads': False,
        # 'write_images': False,
        # 'embeddings_freq': 0,
        # 'embeddings_layer_names': None,
        # 'embeddings_metadata': None
    }
    options.update(kwargs)

    return TensorBoard(**options)


def download_file(source_url, file_name='', force=False, hash_alg='', hash=''):
    if not file_name:
        file_name = source_url.split('/')[-1]
    assert file_name

    from lessdeep.datasets.base import maybe_download

    return maybe_download(file_name, cache_dir('download'), source_url, force,
                          hash_alg, hash)


def clone_model(model, **kwargs):
    '''

    :param model:
    :param kwargs: attribute for all layers such as trainable
    :return:
    '''
    import keras
    if hasattr(model, 'layers'):
        assert isinstance(model, keras.Sequential)
        layers = model.layers
    else:
        layers = model

    def clone_layer(l):
        conf = l.get_config()
        conf.pop('name')
        if l == layers[0] and type(l) != keras.layers.InputLayer:
            conf['input_shape'] = l.input_shape[1:]
        new_layer = type(l).from_config(conf)

        return new_layer

    # TODO: support more than sequential
    new_model = keras.Sequential([clone_layer(l) for l in layers])
    for layer, src_layer in zip(new_model.layers, layers):
        layer.set_weights(src_layer.get_weights())
        for key, val in kwargs.items():
            setattr(layer, key, val)

    return new_model
