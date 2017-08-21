from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from utils.datasets import base
from utils.datasets import default_dir

SOURCE_URL = 'https://www.kaggle.com/c/dogs-vs-cats/download/'


def _extract_data(zip_file_path, extract_dir, stamp):
    def _extract_valid():
        pass

    # Check stamp
    stamp_name = 'extract_' + os.path.basename(zip_file_path)
    extract_tmp = extract_dir + '_tmp'
    if stamp.exists(stamp_name) or not _extract_valid(extract_dir):
        shutil.rmtree(extract_dir)
        shutil.rmtree(extract_tmp)
    elif _extract_valid(extract_dir):
        return

    # Extract file
    base.extract(zip_file_path, extract_tmp)

    # Clean up
    shutil.rmtree(extract_tmp)


def download_data():
    TRAIN_ZIP = 'train.zip'
    TEST_ZIP = 'test1.zip'

    cache_dir = default_dir('kaggle/dog_cat', 'cache')
    stamp_dir = default_dir('kaggle/dog_cat', 'stamp')
    download_dir = os.path.join(cache_dir, 'download')

    stamp = base.FileStamp(stamp_dir)

    # download
    local_file = base.maybe_download(TRAIN_ZIP, download_dir,
                                     SOURCE_URL + TEST_ZIP)

    # extract
    _extract_data(stamp=stamp)