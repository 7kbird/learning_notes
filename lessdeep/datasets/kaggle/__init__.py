import yaml
import os
import sys
import shutil
from mechanicalsoup import Browser
from sklearn.model_selection import train_test_split
from progressbar import ProgressBar

KAGGLE_BROWSER = None


class _Config(object):
    def __init__(self, competition_id):
        config_path = os.path.join(os.path.dirname(__file__), 'kaggle_data.yml')
        with open(config_path) as stream:
            try:
                conf = yaml.load(stream)
            except yaml.YAMLError as err:
                raise RuntimeError('Kaggle datasets config file {0} is '
                                   'corrupted:{1}'.format(config_path, str(err)))
        if competition_id not in conf:
            self._conf = None
        else:
            self._conf = conf[competition_id]

    def is_empty(self):
        return not self._conf

    def train(self):
        from collections import namedtuple
        TrainDataConfig = namedtuple('TrainDataConfig',
                                     ('file', 'root'))

        train_conf = self._conf['train']
        if isinstance(train_conf, str):
            return TrainDataConfig(train_conf, '')
        elif isinstance(train_conf, dict):
            def conf_value(key, default=''):
                return train_conf[key] if key in train_conf else default
            return TrainDataConfig(conf_value('file'),
                                   conf_value('root').strip('/'))
        else:
            raise RuntimeError('Training config is not valid: ' +
                               str(train_conf))


def _browser():
    global KAGGLE_BROWSER
    if not KAGGLE_BROWSER:
        KAGGLE_BROWSER = Browser()
    return KAGGLE_BROWSER


def _select(src_dir, num, random_state=7):
    remain_files = []
    select_files = []

    folders, root_files = next(os.walk(src_dir))[1:]
    for folder in folders:
        files_in_dir = os.listdir(os.path.join(src_dir, folder))

        remain, select = train_test_split(files_in_dir, test_size=num,
                                          random_state=random_state)
        select_files.extend([os.path.join(folder, f) for f in select])
        remain_files.extend([os.path.join(folder, f) for f in remain])
    if not folders:
        remain, select = train_test_split(root_files, test_size=num,
                                          random_state=random_state)
        select_files.extend(select)
        remain_files.extend(remain)

    return select_files, remain_files


class _MoveAfterWork(object):
    def __init__(self, path):
        self._path = path
        self._temp_path = self._path + '_temp'

    def __enter__(self):
        shutil.rmtree(self._temp_path, ignore_errors=True)
        os.makedirs(self._temp_path)
        return self._temp_path

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:
            # Rename the working folder only when no error
            shutil.move(self._temp_path, self._path)

        # do not suppress exception so return False
        return False


def get_train(competition_id, validation=0.0, cache_dir=None):
    """Download/Extract training data and split the validation

    Training data is download and extracted for the competition_id and is split
    into two part training set and validation set. Validation data number is
    set by `validation`.

    NOTE: only image folder can be split into validation currently.

    :param competition_id: the last part of the competition url
    :param validation: size of the validation set. If type is float,
    validation size will be `total_size*validation` else `validation` is
    the validation size for each classes
    :param cache_dir: overwrite the default cache directory
    :return: (training_set, validation_set) if validation is not zero else just
    return the downloaded training set. The return value are the path if its
    a folder containing different classes else return value are pandas frame
    """
    conf = _Config(competition_id)
    if conf.is_empty():
        raise RuntimeError("Competition ID [{0}]is not found in "
                           "data config".format(competition_id))
    conf = conf.train()

    # Download and extract whole training data.
    from lessdeep.datasets.kaggle.downloader import download_dataset
    from lessdeep.datasets.base import extract

    if not cache_dir:
        from lessdeep.datasets import default_dir
        cache_dir = default_dir('kaggle/' + competition_id)

    if validation <= 0:
        # Do not split validation set
        extract_dir = os.path.join(cache_dir, 'train_all')
        train_dir = extract_dir
        valid_dir = ''
    else:
        # Different validation size use different extract folder and will split
        # later
        work_dir = os.path.join(cache_dir, 'train_v{0}'.format(validation))
        extract_dir = os.path.join(work_dir, '_ext')
        train_dir = os.path.join(work_dir, 'train')
        valid_dir = os.path.join(work_dir, 'valid')
        if not os.path.exists(valid_dir) or not os.path.exists(train_dir):
            shutil.rmtree(valid_dir, ignore_errors=True)
            shutil.rmtree(train_dir, ignore_errors=True)
            shutil.rmtree(extract_dir, ignore_errors=True)

    # Download cache is different from extract directory so it will be
    # downloaded only once
    down_file = download_dataset(competition_id, conf.file,
                                 os.path.join(cache_dir, 'down'), _browser())

    if not os.path.exists(extract_dir):
        extract(down_file, extract_dir)

    if validation > 0:
        # Split validation
        src_dir = os.path.join(extract_dir, conf.root)
        if not os.path.exists(valid_dir) or not os.path.exists(train_dir):
            valid_files, train_files = _select(src_dir, validation)
        if not os.path.exists(valid_dir):
            print('Splitting validation dataset')
            with _MoveAfterWork(valid_dir) as tmp_dir:
                sys.stdout.flush()
                bar = ProgressBar()
                for rel_path in bar(valid_files):
                    tgt = os.path.join(tmp_dir, rel_path)
                    os.makedirs(os.path.dirname(tgt), exist_ok=True)
                    shutil.move(os.path.join(src_dir, rel_path), tgt)
                sys.stdout.flush()
        if not os.path.exists(train_dir):
            print('Splitting training dataset')
            with _MoveAfterWork(train_dir) as tmp_dir:
                sys.stdout.flush()
                bar = ProgressBar()
                for rel_path in bar(train_files):
                    tgt = os.path.join(tmp_dir, rel_path)
                    os.makedirs(os.path.dirname(tgt), exist_ok=True)
                    shutil.move(os.path.join(src_dir, rel_path), tgt)
                sys.stdout.flush()

    # TODO: add sample

    if valid_dir:
        return train_dir, valid_dir
    else:
        return train_dir

if __name__ == '__main__':
    print(get_train('the-nature-conservancy-fisheries-monitoring',
                    validation=0.3))
