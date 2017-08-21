import os

DEFAULT_DATA_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                                  '..', '..', 'data'))


def default_dir(name_id, dir_type='cache'):
    global DEFAULT_DATA_ROOT

    data_root = os.path.join(DEFAULT_DATA_ROOT, name_id.replace('/',
                                                                os.path.sep))

    if dir_type == 'cache':
        return os.path.join(data_root, '_cache')
    elif dir_type == 'stamp':
        return os.path.join(data_root, '_stamp')
    elif dir_type == 'data':
        return os.path.join(data_root, 'd')

    raise RuntimeError("Default directory not support dir_type: %s" % dir_type)
