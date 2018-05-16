import os
import torchvision
import torch
import shutil
from yama.util.data import download_url, extract_file
from yama.vision.datasets.storage import Storage


def download_lsun(out_dir, set_name, category='', tag='latest'):
    """
    Args:
        out_dir (string): root directory of download file
        set_name (string): one of ['train', 'val', 'test']
        category (string): if is '', then all category is downloaded, else
            choose one from https://github.com/fyu/lsun/blob/master/category_indices.txt
        tag (string): tag of the dataset
    """
    # TODO: more lsun challenges
    # TODO: file checksum
    url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?tag={tag}' \
          '&category={category}&set={set_name}'.format(**locals())
    if set_name == 'test':
        out_name = 'test_lmdb.zip'
    else:
        out_name = '{category}_{set_name}_lmdb.zip'.format(**locals())
    return download_url(url, out_dir, out_name)


class LSUN(torch.utils.data.Dataset):
    """`LSUN <http://lsun.cs.princeton.edu>`_ dataset.
    Args:
        storage (string or Storage): information of the storage
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
    """
    def __init__(self, storage, classes='train', transform=None,
                 target_transform=None, download=True):
        # Start: Copy from torchvision
        categories = ['bedroom', 'bridge', 'church_outdoor', 'classroom',
                      'conference_room', 'dining_room', 'kitchen',
                      'living_room', 'restaurant', 'tower']
        dset_opts = ['train', 'val', 'test']
        if type(classes) == str and classes in dset_opts:
            if classes == 'test':
                classes = [classes]
            else:
                classes = [c + '_' + classes for c in categories]
        if type(classes) == list:
            for c in classes:
                c_short = c.split('_')
                c_short.pop(len(c_short) - 1)
                c_short = '_'.join(c_short)
                if c_short not in categories:
                    raise(ValueError('Unknown LSUN class: ' + c_short + '.'
                                     'Options are: ' + str(categories)))
                c_short = c.split('_')
                c_short = c_short.pop(len(c_short) - 1)
                if c_short not in dset_opts:
                    raise(ValueError('Unknown postfix: ' + c_short + '.'
                                     'Options are: ' + str(dset_opts)))
        else:
            raise(ValueError('Unknown option for classes'))
        # End: Copy from torchvision

        if isinstance(storage, Storage):
            db_path = storage.root('lsun')
        elif isinstance(storage, str):
            db_path = storage
        else:
            raise ValueError('Unknown strage type: ' + str(type(storage)) + '.'
                             'storage must be an instance of Storage or string')
        self.db_path = db_path

        class_roots = []
        if download:
            for class_name in classes:
                class_split = class_name.split('_')
                if len(class_split) == 1:
                    categories, set_name = '', class_split[0]
                else:
                    categories, set_name = class_split
                download_root = os.path.join(db_path, class_name, '_down')
                down_file, download = download_lsun(download_root, set_name, categories)
                extract_root = os.path.join(db_path, class_name, 'ext')
                if download:
                    # Have to extract again, since download file changed
                    shutil.rmtree(extract_root, ignore_errors=True)
                if not os.path.exists(extract_root):
                    extract_file(down_file, extract_root)
                else:
                    print('{} alreay existed, no need to extract again'.format(extract_root))
                class_roots.append((class_name, extract_root))
        else:
            for class_name in classes:
                class_roots.append((class_name, db_path))

        # for each class, create an LSUNClassDataset
        self.dbs = []
        for c, root in class_roots:
            self.dbs.append(torchvision.datasets.lsun.LSUNClass(
                os.path.join(root, c + '_lmdb'),
                transform=transform))

        self.indices = []
        count = 0
        for db in self.dbs:
            count += len(db)
            self.indices.append(count)

        self.length = count
        self.target_transform = target_transform

# Start: Copy from torchvision
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target) where target is the index of the target category.
        """
        target = 0
        sub = 0
        for ind in self.indices:
            if index < ind:
                break
            target += 1
            sub = ind

        db = self.dbs[target]
        index = index - sub

        if self.target_transform is not None:
            target = self.target_transform(target)

        img, _ = db[index]
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

# End: Copy from torchvision
