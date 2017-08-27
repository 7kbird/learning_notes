import os

__default_cache_root=os.path.realpath(os.path.join(os.path.dirname(__file__), '_cache'))


def cache_dir(name):
    return os.path.join(__default_cache_root, name)
