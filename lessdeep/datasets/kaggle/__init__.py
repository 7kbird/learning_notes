import yaml
import os
import sys
import shutil
from mechanicalsoup import Browser
from sklearn.model_selection import train_test_split
import multiprocessing

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

    def test(self):
        from collections import namedtuple
        TestDataConfig = namedtuple('TestingDataConfig',
                                     ('files', 'root'))
        test_conf = self._conf['test']
        root_dir = ''
        if isinstance(test_conf, str):
            files = [test_conf]
        elif isinstance(test_conf, dict):
            def conf_value(key, default=''):
                return test_conf[key] if key in test_conf else default

            files = conf_value('file')
            if type(files) not in [str, list]:
                raise RuntimeError('Testing config is not valid: ' +
                                   str(test_conf))
            if type(files) is list and \
                    not all([type(f) == str for f in files]):
                raise RuntimeError('Testing config is not valid: ' +
                                   str(test_conf))
            root_dir = conf_value('root').strip('/')
        else:
            raise RuntimeError('Testing config is not valid: ' +
                               str(test_conf))
        return TestDataConfig(files, root_dir)

    def categories(self):
        from collections import namedtuple
        CategoriesConfig = namedtuple('CategoriesConfig',
                                     ('type', 'pattern'))
        if 'categories' in self._conf:
            categories_conf = self._conf['categories']
        if isinstance(categories_conf, str):
            if categories_conf not in ['folder',]:
                raise RuntimeError('categories must be `folder`')

            return CategoriesConfig(categories_conf, '')
        elif isinstance(categories_conf, dict):
            if 're' in categories_conf:
                return CategoriesConfig('re', categories_conf['re'])
            else:
                raise RuntimeError('Unknown categories ' +
                                   str(categories_conf))
        else:
            return CategoriesConfig('folder', '')


def _browser():
    global KAGGLE_BROWSER
    if not KAGGLE_BROWSER:
        KAGGLE_BROWSER = Browser()
    return KAGGLE_BROWSER


def _select(src_dir, num, categories, random_state=7):
    remain_files = []
    select_files = []

    folders, root_files = next(os.walk(src_dir))[1:]
    if categories.type == 'folder':
        for folder in folders:
            files_in_dir = [(os.path.join(folder, f),)*2 for f in
                            os.listdir(os.path.join(src_dir,folder))]

            remain, select = train_test_split(files_in_dir, test_size=num,
                                              random_state=random_state)
            select_files.extend(select)
            remain_files.extend(remain)
        if not folders:
            remain, select = train_test_split(root_files, test_size=num,
                                              random_state=random_state)
            select_files.extend([(f,)*2 for f in select])
            remain_files.extend([(f,)*2 for f in remain])
    elif categories.type == 're':
        import re, warnings
        pattern = re.compile(categories.pattern)
        classed_files = dict()
        for file in root_files:
            match = pattern.match(file)
            if match:
                groups = match.groups()
                if len(groups) < 1:
                    raise RuntimeError('File pattern must contain 1 group '
                                       'used as class name:' +
                                       categories.pattern)
                class_name = groups[0]
                if class_name not in classed_files:
                    classed_files[class_name] = []
                classed_files[class_name].append(file)
            else:
                warnings.warn('Some file not match the defined pattern' +
                              os.path.join(src_dir, file))

        for class_name, files in classed_files.items():
            remain, select = train_test_split(files, test_size=num,
                                              random_state=random_state)
            select_files.extend([(f, os.path.join(class_name, f)) for f in
                                 select])
            remain_files.extend([(f, os.path.join(class_name, f)) for f in
                                 remain])
    else:
        raise RuntimeError('Unsupported categories:' + str(categories))

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


def _move_files(args):
    src_dir, src_path, tgt_dir, tgt_path = args
    tgt = os.path.join(tgt_dir, tgt_path)
    os.makedirs(os.path.dirname(tgt), exist_ok=True)
    shutil.move(os.path.join(src_dir, src_path), tgt)


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
    comp_conf = _Config(competition_id)
    if comp_conf.is_empty():
        raise RuntimeError("Competition ID [{0}]is not found in "
                           "data config".format(competition_id))
    conf = comp_conf.train()

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

    if validation <= 0:
        # No validation data set, just return the extracted training set
        return train_dir

    # Split validation
    src_dir = os.path.join(extract_dir, conf.root)
    trn_mv_dir = train_dir + '_move_tmp'
    val_mv_dir = valid_dir + '_move_tmp'
    if not os.path.exists(valid_dir) or not os.path.exists(train_dir):
        shutil.rmtree(trn_mv_dir, ignore_errors=True)
        shutil.rmtree(val_mv_dir, ignore_errors=True)
        valid_files, train_files = _select(src_dir, validation,
                                           comp_conf.categories())
        for stage, data_dir, files in [('validation', val_mv_dir, valid_files),
                                       ('training', trn_mv_dir, train_files)]:
            if not os.path.exists(data_dir):
                from progressbar import ProgressBar
                with _MoveAfterWork(data_dir) as tmp_dir:
                    sys.stdout.flush()
                    bar = ProgressBar(max_value=len(files))
                    bar.widgets = ['Split ' + stage + ' set'] + \
                                  bar.default_widgets()
                    pool = multiprocessing.Pool()
                    args = [(src_dir, p1, tmp_dir, p2) for p1, p2 in files]
                    try:
                        bar.start(len(files))
                        for i, _ in enumerate(pool.imap_unordered(
                                                _move_files, args)):
                            bar.update(i)
                    except Exception as e:
                        pool.terminate()
                        pool.join()
                        raise e
                    bar.finish()
        shutil.move(trn_mv_dir, train_dir)
        shutil.move(val_mv_dir, valid_dir)
    return train_dir, valid_dir


def get_sample(competition_id, sample=10, cache_dir=None):
    """Download/Extract training data and split the sample

    Sample data is download and extracted for the competition_id and extract
    sample set. Sample data number is set by `sample`.

    NOTE: only image folder can be split into sample currently.

    :param competition_id: the last part of the competition url
    :param sample: size of the sample set. If type is float,
    validation size will be `total_size*sample` else `validation` is
    the validation size for each classes
    :param cache_dir: overwrite the default cache directory
    :return: sample data path
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

    if sample <= 0:
        raise ValueError('Sample number must > 0, or float > 0.0')

    # Different sample size use different extract folder and will split
    # later
    work_dir = os.path.join(cache_dir, 'sample_{0}'.format(sample))
    sample_dir = os.path.join(work_dir, 'data')

    if os.path.exists(sample_dir):
        return sample_dir

    # Download cache is different from extract directory so it will be
    # downloaded only once
    down_file = download_dataset(competition_id, conf.file,
                                 os.path.join(cache_dir, 'down'), _browser())


    # Split sample
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


def get_test(competition_id, test_file=None, cache_dir=None):
    """Download/Extract testing data

    Testing data is download and extracted for the competition_id

    NOTE: only image folder can be split into validation currently.

    :param competition_id: the last part of the competition url
    :param test_file: name of the test file, sometimes there're more than one
    testing data. If no name specified, only first testing data will be
    download
    :param cache_dir: overwrite the default cache directory
    :return: test_data folder
    """
    comp_conf = _Config(competition_id)
    if comp_conf.is_empty():
        raise RuntimeError("Competition ID [{0}]is not found in "
                           "data config".format(competition_id))
    conf = comp_conf.test()

    # Download and extract whole training data.
    from lessdeep.datasets.kaggle.downloader import download_dataset
    from lessdeep.datasets.base import extract

    if not cache_dir:
        from lessdeep.datasets import default_dir
        cache_dir = default_dir('kaggle/' + competition_id)

    if test_file and test_file not in conf.files:
        raise ValueError('Cannot find test file {0} in dataset'.format(
            test_file))
    if not test_file:
        test_file = conf.files[0]

    work_dir = os.path.join(cache_dir, 'test'.format(test_file))
    extract_dir = os.path.join(work_dir, '{0}_ext_tmp'.format(test_file))
    test_dir = os.path.join(work_dir, test_file)
    if not os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
        shutil.rmtree(extract_dir, ignore_errors=True)

    # Download cache is different from extract directory so it will be
    # downloaded only once
    down_file = download_dataset(competition_id, test_file,
                                 os.path.join(cache_dir, 'down'),
                                 _browser())

    if not os.path.exists(test_dir):
        shutil.rmtree(extract_dir, ignore_errors=True)
        extract(down_file, extract_dir)
        src_dir = os.path.join(extract_dir, conf.root)
        # since ImageGenerator require images in folders, we create one
        os.makedirs(test_dir, exist_ok=True)
        shutil.move(src_dir, os.path.join(test_dir, '0'))

    return test_dir


if __name__ == '__main__':
    print(get_train('the-nature-conservancy-fisheries-monitoring',
                    validation=0.3))
