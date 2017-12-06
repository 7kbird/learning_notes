import os
from lessdeep.datasets import default_dir
from lessdeep.datasets.base import extract, maybe_download

base_url = 'http://files.grouplens.org/datasets/movielens/'


def _download_dataset(data_path, hash_path, cache_dir):
    hash_file = maybe_download(hash_path, cache_dir, base_url + hash_path)
    with open(hash_file, 'r') as f:
        line = f.readline()
        alg = line.split(' ')[0].lower()
        hash_str = line.split('=')[-1].strip().lower()
    return maybe_download(data_path, cache_dir, base_url + data_path,
                          hash_alg=alg, hash=hash_str)


def download_data():
    cache_dir = default_dir('grouplens/movielens', 'cache')
    data_dir = default_dir('grouplens/movielens', 'data')

    extract_dir = os.path.join(data_dir, 'ext')
    if not os.path.exists(extract_dir):
        local_file = _download_dataset('ml-20m.zip', 'ml-20m.zip.md5',
                                       cache_dir)
        extract(local_file, extract_dir)

    return extract_dir
