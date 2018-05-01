import hashlib
import os
import shutil
import numpy as np
import urllib3
from six.moves import urllib


def _download_single_thread(url, file_path, stamp_path, force, support_range,
                            file_size=0):
    if os.path.exists(stamp_path) and (force or not support_range):
        os.remove(stamp_path)

    with open(file_path, "wb") as out:
        if file_size:
            out.truncate(file_size)
        # TODO: download with progress or forever
        raise NotImplementedError()


class _PartWorker(object):
    """Own a connection pool, and fetch bytes using `Range` header"""
    def __init__(self, url, start, block_size, total_size):
        self.url = url
        self.start = start
        self.block_size = block_size
        self.total_size = total_size
        self.pool = None

    def __enter__(self):
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        self.pool = urllib3.connection_from_url(self.url)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool:
            self.pool.close()

    def __call__(self, index):
        r_start = self.start + index * self.block_size
        r_end = min(self.total_size - 1,
                    self.start + (index + 1) * self.block_size - 1)

        headers = {'Range': 'bytes={}-{}'.format(r_start, r_end)}
        r = self.pool.request('GET', self.url, headers=headers)
        assert len(r.data) == r_end - r_start + 1

        return r.data, index


def _read_stamp(stamp_path, default_value):
    """return value in stamp or ``default_value``"""
    stamp_old = stamp_path + '.old'
    if not os.path.exists(stamp_path):
        if os.path.exists(stamp_old):
            shutil.move(stamp_old, stamp_path)
        else:
            return default_value
    with open(stamp_path, 'r') as f:
        return type(default_value)(f.read())


def _write_stamp(stamp_path, value):
    stamp_old, stamp_new = [stamp_path + postfix for postfix in ['.old', '.new']]

    # File rename is atomic, cannot be interrupted. So use rename and
    # multiple files to make sure stamp_path is always valid
    with open(stamp_new, 'w') as f:
        f.write(str(value))

    if os.path.exists(stamp_old):
        os.remove(stamp_old)
    if os.path.exists(stamp_path):
        shutil.move(stamp_path, stamp_old)
    shutil.move(stamp_new, stamp_path)
    if os.path.exists(stamp_old):
        os.remove(stamp_old)


def _remove_stamp(stamp_path):
    files = [stamp_path + postfix for postfix in ['', '.old', '.new']]
    for f in files:
        if os.path.exists(f):
            os.remove(f)


def _download_multi_thread(url, file_path, stamp_path, force, file_size, threads):
    block_size = 1024 * 1024   # 1M

    file_mode = 'wb'
    if os.path.exists(file_path):
        file_mode = 'r+b'

    # Get start position from stamp
    if force:
        _remove_stamp(stamp_path)
    start_pos = _read_stamp(stamp_path, 0) if not force else 0
    next_sync_idx = 0

    block_num = int(np.ceil((file_size - start_pos) / block_size))

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from yama.util import tqdm
    with open(file_path, file_mode) as out, \
            _PartWorker(url, start_pos, block_size, file_size) as worker, \
            ThreadPoolExecutor(max_workers=threads) as pool:
        # Pre-allocate file disk space
        out.truncate(file_size)
        out.flush()

        bar_dic = dict(total=file_size, initial=start_pos,
                       unit='MB', unit_scale=True,
                       postfix={'file': os.path.basename(file_path)})
        pending_futures = [pool.submit(worker, i) for i in range(block_num)]
        dirty_index = []
        with tqdm(**bar_dic) as bar:
            for future in as_completed(pending_futures):
                buff, i = future.result()
                out.seek(start_pos + i * block_size)
                out.write(buff)
                out.flush()
                bar.update(len(buff))
                dirty_index.append(i)
                if i == next_sync_idx:
                    # Flush and write to stamp
                    os.fsync(out.fileno())
                    dirty_index.sort()
                    while len(dirty_index) > 0 \
                            and dirty_index[0] == next_sync_idx:
                        next_sync_idx = dirty_index.pop(0) + 1
                    _write_stamp(stamp_path, start_pos + next_sync_idx * block_size)


class HeadRequest(urllib.request.Request):
    def get_method(self):
        return "HEAD"


def download_multi_thread(url, local_path, threads=None, force=False):
    """download using multi-threads

    Args:
        url (string): url of the file
        local_path (string): local path of the downloaded file
        threads (int or None, optional): number of threads used for download.
            if None or 0, threads will be used as many as possible
        force (bool, optional): force download, ignore existed
    """
    from six.moves import urllib
    response = urllib.request.urlopen(HeadRequest(url))

    # Get url properties
    file_size = response.info().get('Content-Length')
    file_size = int(file_size) if file_size else 0
    support_range = response.info().get('Accept-Ranges')
    support_range = support_range and support_range.lower() == 'bytes'

    # Get path ready
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_file = local_path + '_unfinished'
    stamp_file = local_path + '_stamp'

    if not file_size or not support_range or threads == 1:
        if not force and not support_range and os.path.exists(tmp_file):
            print('Cannot resume last download: {url} do not support Ranges.'
                  'Use single thread to download'.format(**locals()))
        _download_single_thread(url, tmp_file, stamp_file, force,
                                support_range, file_size)
        # TODO: use multi-thread when support_range == True and file_size == 0
    else:
        _download_multi_thread(url, tmp_file, stamp_file, force, file_size,
                               threads)
    
    if os.path.exists(local_path):
        os.remove(local_path)
    shutil.move(tmp_file, local_path)
    _remove_stamp(stamp_file)


def _hash_update(hash_name, total_size, data_queue):
    hash_alg = hashlib.new(hash_name)
    count = 0
    while count < total_size:
        # Queue will wait new item
        buff = data_queue.get(block=True, timeout=60)
        hash_alg.update(buff)
        count += len(buff)
    assert count == total_size
    assert data_queue.empty()
    data_queue.put(hash_alg.hexdigest())


def verify_hash(local_path, hash_name, hash_val):
    """ Verify hash using two process, one for read one for process

    Args:
        local_path (string): file path
        hash_name (string): one of the hash supported by hashlib
        hash_val (string): hash hex string(no mater lower or upper case)
    Return
        (bool): hash is valid
    """
    from multiprocessing import Process, Queue
    block_size = 50 * 1024 * 1024
    file_size = os.path.getsize(local_path)
    with open(local_path, 'rb') as f:
        data_queue = Queue()
        worker_process = Process(target=_hash_update,
                                 args=(hash_name, file_size, data_queue))
        size_sum = 0
        buff = f.read(block_size)
        worker_process.start()
        while len(buff) > 0:
            data_queue.put(buff)
            size_sum += len(buff)
            buff = f.read(block_size)
        if size_sum != file_size:
            worker_process.terminate()
            raise RuntimeError('File {local_path} read failed: expect size of '
                               '{file_size} bytes, but got {size_sum}'.format(**locals()))
        worker_process.join()   # wait hash update finish

        if worker_process.exitcode != 0:
            raise RuntimeError('Verify hash code failed due to child process error')

        # result is returned as item in queue
        actual_hash = data_queue.get_nowait()
        if type(actual_hash) != str:
            raise RuntimeError('Worker process not terminate right')
        assert data_queue.empty()   # hash should be the last item

        return actual_hash.lower() == hash_val.lower()


def download_url(url, root, filename, force=False, threads=None, **kwargs):
    """Download url with multi-thread(if possible) and verify

    Perform a HTTP(S) download with multi-thread. Download will
    be resumed if last not finished when ``force`` is False

    Args:
        url (string): HTTP(S) url to download
        root (string): Root directory to download file
        filename (string): File name
        force (bool, optinal): Force re-download
        kwargs: additional hash code check, support all hash algorithm that
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
            if not verify_hash(local_file, hash_name, kwargs[hash_name]):
                os.remove(local_file)
        if os.path.exists(local_file):
            return local_file, False
    
    # Download file
    download_multi_thread(url, local_file, force=force, threads=threads)

    # Verify hash code
    if kwargs and not verify_hash(local_file, hash_name, kwargs[hash_name]):
        raise RuntimeError('Download failed: hash not match. '
                           'Expect {0}=\'{1}\''.format(hash_name, kwargs[hash_name]))

    return local_file, True
