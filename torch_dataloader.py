from abc import abstractmethod
from tqdm import tqdm
import torch
import hashlib
import numpy as np
from os import makedirs, listdir
from os.path import join, exists, dirname, abspath, splitext
from torch.utils.data import Dataset

def get_hash(x: str):
    """Generate a hash from a string.
    """
    h = hashlib.md5(x.encode())
    return h.hexdigest()


def make_dir(folder_name):
    if not exists(folder_name):
        makedirs(folder_name)

class Cache(object):

    def __init__(self, func, cache_dir: str, cache_key: str):
        self.func = func
        self.cache_dir = join(cache_dir, cache_key)
        make_dir(self.cache_dir)
        self.cached_ids = [splitext(p)[0] for p in listdir(self.cache_dir)]

    def __call__(self, unique_id: str, *data):
        fpath = join(self.cache_dir, str('{}.npy'.format(unique_id)))

        if not exists(fpath):
            output = self.func(*data)

            self._write(output, fpath)
            self.cached_ids.append(unique_id)
        else:
            output = self._read(fpath)

        return self._read(fpath)

    def _write(self, x, fpath):
        np.save(fpath, x)
        # tmp = np.load(fpath, allow_pickle=True)

    def _read(self, fpath):
        return np.load(fpath, allow_pickle=True).item()


class TorchDataloader(Dataset):
    def __init__(self,
                 dataset=None,
                 preprocess=None,
                 transform=None,
                 use_cache=True,
                 steps_per_epoch=None,
                 **kwargs):
        self.dataset = dataset
        self.preprocess = preprocess
        self.steps_per_epoch = steps_per_epoch

        if preprocess is not None and use_cache:
            cache_dir = getattr(dataset.cfg, 'cache_dir')
            assert cache_dir is not None, 'cache directory is not given'

            cache_dir = join(dataset.cfg.logs_dir, cache_dir)
            make_dir(cache_dir)
            
            self.cache_convert = Cache(preprocess,
                                       cache_dir=cache_dir,
                                       cache_key=get_hash(repr(preprocess)))

            uncached = [
                idx for idx in range(len(dataset)) if dataset.get_attr(idx)['name'] 
                not in self.cache_convert.cached_ids
            ]
            if len(uncached) > 0:
                for idx in tqdm(range(len(dataset)), desc='preprocess'):
                    attr = dataset.get_attr(idx)
                    name = attr['name']
                    if name in self.cache_convert.cached_ids:
                        continue
                    data = dataset.get_data(idx)

                    self.cache_convert(name, data, attr)

        else:
            self.cache_convert = None

        self.transform = transform

    def __getitem__(self, index):
        """Returns the item at index idx. """
        dataset = self.dataset
        index = index % len(dataset)

        attr = dataset.get_attr(index)
        if self.cache_convert:
            data = self.cache_convert(attr['name'])
        elif self.preprocess:
            data = self.preprocess(dataset.get_data(index), attr)
        else:
            data = dataset.get_data(index)

        if self.transform is not None:
            data = self.transform(data, attr)

        inputs = {'data': data, 'attr': attr}

        return inputs

    def __len__(self):
        if self.steps_per_epoch is not None:
            steps_per_epoch = self.steps_per_epoch
        else:
            steps_per_epoch = len(self.dataset)
        return steps_per_epoch
