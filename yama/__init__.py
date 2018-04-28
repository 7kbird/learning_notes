import os


class _Config(object):
    def __init__(self):
        base_dir = os.path.expanduser('~')
        if not os.access(base_dir, os.W_OK):
            base_dir = '/tmp'
        self.cache_dir = os.path.join(base_dir, '.yama')


config = _Config()
import torch
torch.autograd.Variable().zero_()
import torchvision
torchvision.datasets.LSUN