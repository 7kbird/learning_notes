from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ...datasets import base
from ...datasets import default_dir
from .downloader import download_extract

import os
from mechanicalsoup import Browser


class Downloaded(object):
    def __init__(self, train_path, test_path):
        self._train_path = train_path
        self._test_path = test_path

    def load(self, validate=1000, sample=10):
        return 'train', 'validate', 'sample'


def download_data():
    cache_dir = default_dir('kaggle/porto_seguro_safe_driver', 'cache')
    stamp_dir = default_dir('kaggle/porto_seguro_safe_driver', 'stamp')
    data_dir = default_dir('kaggle/porto_seguro_safe_driver', 'data')

    stamp = base.FileStamp(stamp_dir)

    browser = Browser()
    train_dir = download_extract('porto-seguro-safe-driver-prediction',
                                 'train.zip', download_dir=cache_dir,
                                  browser=browser, extract_root=data_dir)

