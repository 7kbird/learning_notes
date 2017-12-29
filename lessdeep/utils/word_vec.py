import os
from lessdeep.datasets.base import maybe_download
import zipfile


def __default_dir(name_id):
    data_root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                              '..', '..', '_vec'))

    return os.path.join(data_root, name_id.replace('/', os.path.sep))


def glove(name='6B', sub='50d'):
    all_vectors = {
        '6B': ('glove.6B', ('50d', '100d', '200d', '300d'),
               '056ea991adb4740ac6bf1b6d9b50408b'),
        '42B': ('glove.42B.300d', (), '8d2b60d9970a8717cdfc0f6834b06f2e'),
    }
    if name.upper() not in all_vectors:
        raise RuntimeError(name + " no valid glove name: choose from: " +
                           ', '.join(all_vectors.keys()))
    down_base_name, sub_names, md5sum = all_vectors[name.upper()]
    if sub.lower() not in sub_names and len(sub_names) != 0:
        raise RuntimeError(sub + " no valid glove data: choose from: " +
                           ', '.join(sub_names))
    sub = sub.lower()
    name = name.lower()
    down_name = down_base_name + '.zip'
    if not sub_names:
        file_name = down_base_name + '.txt'
    else:
        file_name = down_base_name + '.' + sub.lower() + '.txt'

    cache_dir = __default_dir('glove/' + name + '/cache')
    data_dir = __default_dir('glove/' + name + '/d')

    extract_dir = os.path.join(data_dir, 'ext')

    down_file = os.path.join(cache_dir, down_name)
    if not os.path.exists(down_file):
        base_url = 'http://nlp.stanford.edu/data/'
        maybe_download(down_name, cache_dir, base_url + down_name,
                       hash_alg='md5', hash=md5sum)

    if zipfile.is_zipfile(down_file):
        zf = zipfile.ZipFile(down_file, 'r')
        file_bytes = zf.read(file_name)
        zf.close()
    else:
        raise RuntimeError("Download file is not valid zip: " + down_file)

    vocabulary = []
    word_vectors = []
    for i, line in enumerate(file_bytes.splitlines()):
        comp = line.rstrip().split()
        w, vec = comp[0].decode(), list(map(float, comp[1:]))
        vocabulary.append(w)
        word_vectors.append(vec)

    return vocabulary, word_vectors
