import urllib.request
import re

_word2vec_readme_url = 'https://raw.githubusercontent.com/Kyubyong/' \
                       'wordvectors/master/README.md'


def get_embedding_url(language, embedding_type):
    """Get Downloading url for embeddings

    :param language: see https://github.com/Kyubyong/wordvectors
    :param embedding_type: 'word2vec' or 'fasttext'
    :return: url of the emdedding
    """
    if embedding_type not in ['word2vec', 'fasttext']:
        raise ValueError('embedding_type only support word2vec, fasttext')

    # Get all supported language and downloading urls from README.md
    nonlocal _word2vec_readme_url
    response = urllib.request.urlopen(_word2vec_readme_url)
    readme = response.read().decode('utf-8')
    if embedding_type == 'word2vec':
        url_pairs = re.findall(r'\|\[(\w[\w\s]+\w)\s*\(w\)\]\((https://['
                               r'\w\d\./\?=]+)\)[^\|]*\|', readme)
    else:
        url_pairs = re.findall(r'\|\s*\[(\w[\w\s]+\w)\s*\(f\)\]\(('
                               r'https://[\w\d\./\?=]+)\)[^\|]*\|', readme)

    url_dic = {name.lower(): url for name, url in url_pairs}
    if language.lower() not in url_dic.keys():
        raise ValueError('Language {0} not supported, choose from '
                         '{1}'.format(language, list(url_dic.keys())))

    return url_dic[language.lower()]

