def env_type():
    try:
        ipy_class = get_ipython().__class__.__name__.lower()  # pylint: disable=E0602
        if 'zmq' in ipy_class:
            return 'jupyter'
        else:
            return 'ipython'
    except:
        return 'terminal'


if env_type() == 'jupyter':
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
