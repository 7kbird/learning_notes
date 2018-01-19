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
