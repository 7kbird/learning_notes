from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from ...datasets import base
from ...datasets import default_dir
from .downloader import download_extract

from mechanicalsoup import Browser


class Downloaded(object):
    def __init__(self, train_path, test_path, split_dir):
        self._train_path = train_path
        self._test_path = test_path
        self._split_dir = split_dir

    def load(self, validate=500, sample=10):
        #return 'train', 'validate', 'sample'


def download_data():
    cache_dir = default_dir('kaggle/fishery_monitor', 'cache')
    data_dir = default_dir('kaggle/fishery_monitor', 'data')

    browser = Browser()
    train_dir = download_extract('porto-seguro-safe-driver-prediction',
                                 'train.7z', download_dir=cache_dir,
                                  browser=browser, extract_root=data_dir)

    test_dir = download_extract('porto-seguro-safe-driver-prediction',
                                 'test.7z', download_dir=cache_dir,
                                 browser=browser, extract_root=data_dir)

    split_dir = os.path.join(data_dir, 's')
    return Downloaded(train_dir, test_dir, split_dir)
