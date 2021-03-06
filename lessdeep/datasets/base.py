# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Base utilities for loading datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import sys
import os
from os import path
import random
import time
import tarfile
import zipfile
import shutil
from glob import glob

import numpy as np
from six.moves import urllib
import hashlib

from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_csv_with_header(filename,
                         target_dtype,
                         features_dtype,
                         target_column=-1):
    """Load dataset from CSV file with a header row."""
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        header = next(data_file)
        n_samples = int(header[0])
        n_features = int(header[1])
        data = np.zeros((n_samples, n_features), dtype=features_dtype)
        target = np.zeros((n_samples,), dtype=target_dtype)
        for i, row in enumerate(data_file):
            target[i] = np.asarray(row.pop(target_column), dtype=target_dtype)
            data[i] = np.asarray(row, dtype=features_dtype)

    return Dataset(data=data, target=target)


def load_csv_without_header(filename,
                            target_dtype,
                            features_dtype,
                            target_column=-1):
    """Load dataset from CSV file without a header row."""
    with gfile.Open(filename) as csv_file:
        data_file = csv.reader(csv_file)
        data, target = [], []
        for row in data_file:
            target.append(row.pop(target_column))
            data.append(np.asarray(row, dtype=features_dtype))

    target = np.array(target, dtype=target_dtype)
    data = np.array(data)
    return Dataset(data=data, target=target)


def shrink_csv(filename, ratio):
    """Create a smaller dataset of only 1/ratio of original data."""
    filename_small = filename.replace('.', '_small.')
    with gfile.Open(filename_small, 'w') as csv_file_small:
        writer = csv.writer(csv_file_small)
        with gfile.Open(filename) as csv_file:
            reader = csv.reader(csv_file)
            i = 0
            for row in reader:
                if i % ratio == 0:
                    writer.writerow(row)
                i += 1


def load_iris(data_path=None):
    """Load Iris dataset.

    Args:
        data_path: string, path to iris dataset (optional)

    Returns:
      Dataset object containing data in-memory.
    """
    if data_path is None:
        module_path = path.dirname(__file__)
        data_path = path.join(module_path, 'data', 'iris.csv')
    return load_csv_with_header(
        data_path,
        target_dtype=np.int,
        features_dtype=np.float)


def load_boston(data_path=None):
    """Load Boston housing dataset.

    Args:
        data_path: string, path to boston dataset (optional)

    Returns:
      Dataset object containing data in-memory.
    """
    if data_path is None:
        module_path = path.dirname(__file__)
        data_path = path.join(module_path, 'data', 'boston_house_prices.csv')
    return load_csv_with_header(
        data_path,
        target_dtype=np.float,
        features_dtype=np.float)


def retry(initial_delay,
          max_delay,
          factor=2.0,
          jitter=0.25,
          is_retriable=None):
    """Simple decorator for wrapping retriable functions.

    Args:
      initial_delay: the initial delay.
      factor: each subsequent retry, the delay is multiplied by this value.
          (must be >= 1).
      jitter: to avoid lockstep, the returned delay is multiplied by a random
          number between (1-jitter) and (1+jitter). To add a 20% jitter, set
          jitter = 0.2. Must be < 1.
      max_delay: the maximum delay allowed (actual max is
          max_delay * (1 + jitter).
      is_retriable: (optional) a function that takes an Exception as an argument
          and returns true if retry should be applied.
    """
    if factor < 1:
        raise ValueError('factor must be >= 1; was %f' % (factor,))

    if jitter >= 1:
        raise ValueError('jitter must be < 1; was %f' % (jitter,))

    # Generator to compute the individual delays
    def delays():
        delay = initial_delay
        while delay <= max_delay:
            yield delay * random.uniform(1 - jitter, 1 + jitter)
            delay *= factor

    def wrap(fn):
        """Wrapper function factory invoked by decorator magic."""

        def wrapped_fn(*args, **kwargs):
            """The actual wrapper function that applies the retry logic."""
            for delay in delays():
                try:
                    return fn(*args, **kwargs)
                except Exception as e:  # pylint: disable=broad-except)
                    if is_retriable is None:
                        continue

                    if is_retriable(e):
                        time.sleep(delay)
                    else:
                        raise
            return fn(*args, **kwargs)

        return wrapped_fn

    return wrap


_RETRIABLE_ERRNOS = {
    110,  # Connection timed out [socket.py]
}


def _is_retriable(e):
    return isinstance(e, IOError) and e.errno in _RETRIABLE_ERRNOS


@retry(initial_delay=1.0, max_delay=16.0, is_retriable=_is_retriable)
def urlretrieve_with_retry(url, filename=None):
    return urllib.request.urlretrieve(url, filename)


def hash_check(filepath, hash_alg, hash):
    hash_obj = hashlib.__dict__[hash_alg]()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest().lower() == hash.lower()


def maybe_download(filename, work_directory, source_url, force=False,
                   hash_alg='', hash=''):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.
        force: force download
        hash_alg: algorithm of hash
        hash: hash string
    Returns:
        Path to resulting file.
    """
    if not gfile.Exists(work_directory):
        gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)

    if gfile.Exists(filepath) and not force and hash_alg:
        if not hash_check(filepath, hash_alg, hash):
            force = True

    if not gfile.Exists(filepath) or force:
        print('Downloading', source_url)
        temp_filename = os.path.join(work_directory,
                                     source_url.split('/')[-1] + '.downtmp')
        temp_file_name, _ = urlretrieve_with_retry(source_url, temp_filename)
        if hash_alg:
            if not hash_check(temp_file_name, hash_alg, hash):
                raise "Downloaded hash not match"
        gfile.Rename(temp_file_name, filepath, overwrite=True)
        with gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def extract(filename, data_root, remove_single=True):
    print('Extracting data for %s. This may take a while. Please wait.' % data_root)
    shutil.rmtree(data_root, ignore_errors=True)

    extract_tmp = os.path.join(data_root + '_extract_unfinished')

    if tarfile.is_tarfile(filename):
        tar = tarfile.open(filename, bufsize=1024*1024*20)
        sys.stdout.flush()
        tar.extractall(extract_tmp)
        tar.close()
    elif zipfile.is_zipfile(filename):
        zf = zipfile.ZipFile(filename, 'r')
        zf.extractall(extract_tmp)
        zf.close()
    else:
        raise NotImplementedError('File type is not supported for extraction: %s' % filename)

    if remove_single and len(os.listdir(extract_tmp)) == 1:
        sub_dir_name = os.listdir(extract_tmp)[0]
        os.chmod(extract_tmp, 0o777)
        os.rename(os.path.join(extract_tmp, sub_dir_name), data_root)
        os.rmdir(extract_tmp)
    else:
        os.rename(extract_tmp, data_root)


def extract_members(filename, remove_single=True):
    # TODO: remove single
    if tarfile.is_tarfile(filename):
        tar = tarfile.open(filename, bufsize=1024*1024*20)
        file_list = tar.getnames()
        tar.close()
    elif zipfile.is_zipfile(filename):
        zf = zipfile.ZipFile(filename, 'r')
        file_list = zf.namelist()
        zf.close()
    else:
        raise NotImplementedError(
            'File type is not supported for extraction: %s' % filename)

    return file_list


class FileStamp(object):
    def __init__(self, root_dir):
        self.root = root_dir

    def _file(self, name):
        return os.path.join(self.root, 'stamp_%s' % name)

    def exists(self, name):
        #return os.path.exists(self._file(name))
        return len(glob(self._file(name))) > 0

    def create(self, name):
        file_path = self._file(name)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        with open(file_path, 'w') as f:
            pass

        class StampHolder(object):
            def __init__(self, parent, name):
                self.stamp = parent
                self.name = name

            def remove(self):
                self.stamp.remove(self.name)

        return StampHolder(self, name)

    def remove(self, name):
        for f in glob(self._file(name)):
            os.remove(f)
