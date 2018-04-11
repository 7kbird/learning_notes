import os
import yama


def maybe_download():
    pass

import keras.applications.vgg16
def download_file(source_url, file_name='', force=False, hash_alg='', hash=''):
    if not file_name:
        file_name = source_url.split('/')[-1]
    assert file_name
    if hash:
        file_name = hash + '_' + file_name

    file_path = os.path.join(yama._yama_config.cache_dir, file_name)

    return maybe_download(file_name, cache_dir('download'), source_url, force,
                          hash_alg, hash)