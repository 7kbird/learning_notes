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
