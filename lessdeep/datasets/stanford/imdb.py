import os
import re
import pickle
from lessdeep.datasets import default_dir
from lessdeep.datasets.base import extract, maybe_download

# from http://ai.stanford.edu/~amaas/data/sentiment/
base_url = 'http://ai.stanford.edu/~amaas/data/sentiment/'


def download_data():
    cache_dir = default_dir('stanford/imdb', 'cache')
    data_dir = default_dir('stanford/imdb', 'data')

    extract_dir = os.path.join(data_dir, 'ext')
    if not os.path.exists(extract_dir):
        file_name = 'aclImdb_v1.tar.gz'
        local_file = maybe_download(file_name, cache_dir, base_url + file_name)
        extract(local_file, extract_dir)

        # remove .feat files to save disk
        from glob import glob
        for file in glob(os.path.join(extract_dir, 'train', '*.feat')):
            os.remove(file)
        for file in glob(os.path.join(extract_dir, 'test', '*.feat')):
            os.remove(file)

    return extract_dir


def _load_sentences(file_dir):
    for root, dirs, files in os.walk(file_dir):
        txt_files = sorted(files, key=lambda s: int(s.split('_')[0]))
        break
    contents = []
    for filename in txt_files:
        with open(os.path.join(file_dir, filename), encoding="utf8") as f:
            contents.append(f.read())
    return contents


def _clean_str(s):
    s = re.sub(r"[^A-Za-z0-9\']", " ", s)
    s = re.sub(r"(\'s|\'ve|n\'t|\'re|\'d|\'ll) ", r" \1 ", s)
    s = re.sub(r"\'(?!s |ve |t )", " ", s)
    s = re.sub(r"(\w)\1{2,}", r'\1', s)     # repeated char
    s = re.sub(r" (\d+)(s|th)", r" \1 \2 ", s)  # 100th 1970s

    # remove plural
    s = re.sub(r'\s([A-Za-z]+[sxz]|[A-Za-z]+[^aeioudgkprt]h)es\s', r" \1 s ", s)
    s = re.sub(r'\s([A-Za-z]+[^aeiou])ies\s', r" \1y s ", s)
    s = re.sub(r'([A-Za-z]{2,}[^aeious])s\s', r"\1 s ", s)

    # remove ed
    s = re.sub(r'([A-Za-z]{2,}[i])ed\s', r"\1y ed ", s)
    s = re.sub(r"([A-Za-z]{2,})([gmnpr]){2}ed\s", r"\1\2 ed ", s)
    s = re.sub(r'([A-Za-z]{2,}[cgsvr]e)d\s', r"\1 ed ", s)
    s = re.sub(r'([A-Za-z]{2,}[htnpx])ed\s', r"\1 ed ", s)

    # TODO: settled, filled, homed, armed, liked, asked, wanted, united, unwed

    # clear blanks
    s = re.sub(r"\s{2,}", " ", s)

    return s.strip().lower()


def get_words():
    root_dir = download_data()

    vocab_file = os.path.join(root_dir, '_vocab.txt')
    if os.path.exists(vocab_file):
        print('Use pre-calculated vocabulary')
        with open(vocab_file, encoding="utf8") as f:
            return f.read().strip().splitlines()

    import itertools

    def _load_words(file_dir):
        sentences = _load_sentences(file_dir)
        return itertools.chain(*[_clean_str(s).split(' ') for s in sentences])

    train_neg = _load_words(os.path.join(root_dir, 'train', 'neg'))
    train_pos = _load_words(os.path.join(root_dir, 'train', 'pos'))
    test_neg = _load_words(os.path.join(root_dir, 'test', 'neg'))
    test_pos = _load_words(os.path.join(root_dir, 'test', 'pos'))

    from collections import Counter

    # ordered by mostly used
    word_counts = Counter(itertools.chain(train_neg, train_pos, test_neg,
                                          test_pos))
    vocab = [v for v, cnt in word_counts.most_common()]

    vocab_tmp = vocab_file + '.tmp'
    with open(vocab_tmp, 'w', encoding="utf8") as f:
        for v in vocab:
            f.write(v)
            f.write('\n')
    os.rename(vocab_tmp, vocab_file)
    return vocab


def get_word_index():
    words = get_words()
    return dict({name: i for i, name in enumerate(words)})


def load_data():
    root_dir = download_data()

    cache_file = os.path.join(root_dir, '_datasets.pkl')

    if os.path.exists(cache_file):
        print('Use pre-calculated data')
        return pickle.load(open(cache_file, 'rb'))

    work2index = get_word_index()

    def _str_to_idx(s):
        words = s.split(' ')
        return [work2index[w] for w in words]

    def _load_files(file_dir):
        sentences = _load_sentences(file_dir)

        return [_str_to_idx(_clean_str(s)) for s in sentences]

    train_neg = _load_files(os.path.join(root_dir, 'train', 'neg'))
    train_pos = _load_files(os.path.join(root_dir, 'train', 'pos'))
    test_neg = _load_files(os.path.join(root_dir, 'test', 'neg'))
    test_pos = _load_files(os.path.join(root_dir, 'test', 'pos'))

    train_x = train_pos + train_neg
    train_y = [1]*len(train_pos) + [0]*len(train_neg)

    test_x = test_pos + test_neg
    test_y = [1]*len(test_pos) + [0]*len(test_neg)
    res = (train_x, train_y), (test_x, test_y)

    pickle.dump(res, open(cache_file, 'wb'))

    return res
