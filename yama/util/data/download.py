import hashlib
import os
import shutil

def _download_single_thread(url, file_path, stamp_path, force, file_size=0):
    with open(file_path, "wb") as out:
        if file_size:
            out.truncate(file_size)

def _download_range(req, start, end):
    from six.moves import urllib
    req = urllib.request.Request(url)
    req.headers['Range'] = 'bytes=%s-%s' % (start, end)
    f = urllib.request.urlopen(req)
    # https://stackoverflow.com/questions/5666576/show-the-progress-of-a-python-multiprocessing-pool-map-call


def _download_multi_thread(url, file_path, stamp_path, force, file_size):
    from six.moves import urllib
    block_size = 10 * 1024 * 1024   # 10M
    with open(file_path, "wb") as out:
        out.truncate(file_size)

def download_multi_thread(url, local_path, force=False):
    from six.moves import urllib
    response = urllib.request.urlopen(url)

    # Get url properties
    file_size = response.info().get('Content-Length')
    file_size = int(file_size) if file_size else 0
    support_range = response.info().get('Accept-Ranges')
    support_range = support_range and support_range.lower() == 'bytes'

    # Get path ready
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_file = local_path + '_unfinished'
    stamp_file = local_path + '_stamp'

    if not file_size:
        print('{url} cannot get file Content-Length: {file_size}.'
              'Use single thread to download'.format(**locals()))
        _download_single_thread(url, tmp_file, stamp_file, force)
    elif not support_range:
        if os.path.exists(tmp_file):
            print('Cannot resume last download: {url} do not support Ranges.'
                  'Use single thread to download'.format(**locals()))
        _download_single_thread(url, tmp_file, stamp_file, force=True,
                                file_size=file_size)
    else:
        _download_multi_thread(url, tmp_file, stamp_file, force, file_size)
    
    if os.path.exists(local_path):
        os.remove(local_path)
    shutil.move(tmp_file, local_path)


def varify_hash(local_path, hash_name, hash_val):
    hash_alg = hashlib.new(hash_name)
    

def download_url(url, root, filename, force=False, **kwargs):
    """Download url with mutli-thread(if possible) and varify

    Perform a HTTP(S) download with multi-thread. Download will
    be resumed if last not finished when ``force`` is False

    Args:
        url (string): HTTP(S) url to download
        root (string): Root directory to download file
        filename (string): File name
        force (bool, optinal): Force re-download
        kwargs: additional hash code check, support all hash algorith that
                hashlib supports. e.g. `md5='1234567890abcdef'`
    Return:
        tuple of (file_path, download): the downloaded file path and a bool
            download flag, True means not ``force`` and already exists.
    """
    # Get Hash code
    if kwargs:
        hash_name = list(kwargs.keys())[0]
        if len(kwargs) != 1 or hash_name not in hashlib.algorithms_available:
            raise ValueError('hash not supported', kwargs)
    
    # Check exists first
    local_file = os.path.join(root, filename)
    if not force and os.path.exists(local_file):
        if kwargs:
            if not varify_hash(local_file, hash_name, kwargs[hash_name]):
                os.remove(local_file)
        if os.path.exists(local_file):
            return local_file, False
    
    # Download file
    download_multi_thread(url, local_file, force=force)

    # Verify hash code
    if kwargs and not varify_hash(local_file, hash_name, kwargs[hash_name]):
        raise RuntimeError('Download failed: hash not match. '
                           'Expect {0}=\'{1}\''.format(hash_name, kwargs[hash_name]))

    return local_file, True
