import os


class Storage(object):
    """An abstract class represent an image storage

    All image storage should subclass it. All subclasses should override
    ``root``, that provide the storage path for the source
    """
    def __init__(self):
        pass

    def root(self, source_name):
        raise NotImplementedError()


class PaperSpaceGradientStrage(Storage):
    """Image storage for Paperespage Gradient Storage

    Paperespage Gradient provide a default readonly directory which contains
    most popular datasets.
    """
    def __init__(self, download_dir='/storage/down', default_dir='/datasets'):
        self.download_dir = download_dir
        self.default_dir = default_dir
    
    def root(self, source_name):
        support_dic = {
            'imagenet_full': 'ImageNetFull',
        }
        if source_name in support_dic:
            return os.path.join(self.default_dir, support_dic[source_name])
        return os.path.join(self.download_dir, source_name)


class LocalStorage(Storage):
    def __init__(self, root_dir):
        self.root_dir = root_dir
    
    def root(self, source_name):
        return os.path.join(self.root_dir, source_name)
