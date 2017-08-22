from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob
import shutil

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from utils.datasets import base
from utils.datasets import default_dir

SOURCE_URL = 'https://www.kaggle.com/c/dogs-vs-cats/download/'


def _extract_data(zip_file_path, extract_dir, stamp, force=False):
    def _extract_valid(dir_name):
        if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
            return False
        sub_dirs = os.listdir(dir_name)
        return 'dogs' in sub_dirs and 'cats' in sub_dirs

    # Check stamp
    stamp_name = 'extract_' + os.path.basename(zip_file_path)
    extract_tmp = extract_dir + '_tmp'
    if force or stamp.exists(stamp_name) or not _extract_valid(extract_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
        shutil.rmtree(extract_tmp, ignore_errors=True)
    elif _extract_valid(extract_dir):
        return

    stamp.create(stamp_name)
    # Extract file
    base.extract(zip_file_path, extract_tmp)

    # Reorder file
    os.makedirs(os.path.join(extract_dir, '_dogs'))
    os.makedirs(os.path.join(extract_dir, '_cats'))

    dog_files = glob(os.path.join(extract_tmp, 'dog.') + '*')
    cat_files = glob(os.path.join(extract_tmp, 'cat.') + '*')

    for f in dog_files:
        shutil.move(f, os.path.join(extract_dir, '_dogs'))
    for f in cat_files:
        shutil.move(f, os.path.join(extract_dir, '_cats'))

    os.rename(os.path.join(extract_dir, '_dogs'), os.path.join(extract_dir, 'dogs'))
    os.rename(os.path.join(extract_dir, '_cats'), os.path.join(extract_dir, 'cats'))

    # Clean up
    shutil.rmtree(extract_tmp)
    stamp.remove(stamp_name)


def _stamp_download(filename, download_dir, url, stamp):
    stamp_name = 'download_' + filename
    force = False
    if stamp.exists(stamp_name):
        force = True

    stamp.create(stamp_name)
    local_file = base.maybe_download(filename, download_dir, url, force=force)
    stamp.remove(stamp_name)

    return local_file


def _split_validate(src_dir, valid_num):
    tgt_dir = os.path.realpath(os.path.join(src_dir, '..', 'valid'))
    shutil.rmtree(tgt_dir, ignore_errors=True)

    for fold in ['cats', 'dogs']:
        rand_files = np.random.choice(os.listdir(os.path.join(src_dir, fold)), valid_num, replace=False)
        move_tgt = os.path.join(tgt_dir, fold)
        os.makedirs(move_tgt)

        for f in rand_files:
            shutil.move(os.path.join(src_dir, fold, f), move_tgt)


def _select_samples(src_dir, sample_num):
    tgt_dir = os.path.realpath(os.path.join(src_dir, '..', 'sample'))
    shutil.rmtree(tgt_dir, ignore_errors=True)

    sample_valid = max(sample_num//10, 4)

    for fold in ['cats', 'dogs']:
        all_imgs = os.listdir(os.path.join(src_dir, fold))

        # Random train samples
        rand_files = np.random.choice(all_imgs, sample_num, replace=False)
        copy_tgt = os.path.join(tgt_dir, 'train', fold)
        os.makedirs(copy_tgt)

        for f in rand_files:
            shutil.copy(os.path.join(src_dir, fold, f), copy_tgt)

        # Random validate samples
        rand_valid = np.random.choice(
            [f for f in all_imgs if f not in rand_files], sample_valid,
            replace=False)

        copy_tgt = os.path.join(tgt_dir, 'valid', fold)
        os.makedirs(copy_tgt)
        for f in rand_valid:
            shutil.copy(os.path.join(src_dir, fold, f), copy_tgt)


def download_data(validate=1000, sample=10):
    train_zip = 'train.zip'
    test_zip = 'test1.zip'

    cache_dir = default_dir('kaggle/dog_cat', 'cache')
    stamp_dir = default_dir('kaggle/dog_cat', 'stamp')
    data_dir = default_dir('kaggle/dog_cat', 'data')

    stamp = base.FileStamp(stamp_dir)

    # train data
    local_file = _stamp_download(train_zip, cache_dir, SOURCE_URL + train_zip, stamp)
    force_extract_train = not stamp.exists('valid_%d' % validate) or not os.path.exists(os.path.join(data_dir, 'valid'))
    _extract_data(local_file, os.path.join(data_dir, 'train'), stamp=stamp, force=force_extract_train)

    # test data
    local_file = _stamp_download(test_zip, cache_dir, SOURCE_URL + test_zip, stamp)
    _extract_data(local_file, os.path.join(data_dir, 'test1'), stamp=stamp)

    # validate data
    stamp.remove('valid_*')
    if force_extract_train:
        _split_validate(os.path.join(data_dir, 'train'), validate)
    stamp.create('valid_%d' % validate)

    # sample data
    if not stamp.exists('sample_%d' % sample) or not os.path.exists(
            os.path.join(data_dir, 'sample')):
        stamp.remove('sample_*')
        _select_samples(os.path.join(data_dir, 'train'), sample)
        stamp.create('sample_%d' % sample)

    return data_dir
