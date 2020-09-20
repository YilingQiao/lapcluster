import numpy as np
import pandas as pd
import os, sys, glob, pickle

from pathlib import Path
from os.path import join, exists, dirname, abspath, splitext
import random
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from tqdm import tqdm
import logging
from laplacian import load_obj
from utils import construct_edges

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class Coseg:
    """
    Toronto3D dataset, used in visualizer, training, or test
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.dataset_path = cfg.dataset_path

        self.train_files = []
        self.test_files = []

        train_path = join(self.dataset_path, 'train')
        test_path = join(self.dataset_path, 'test')
        self.train_files += [ join(train_path, p) for p in os.listdir(train_path)]
        self.test_files += [ join(test_path, p) for p in os.listdir(test_path)]

    def get_split(self, split):
        return CosegSplit(self, split=split)

    def get_split_list(self, split):
        if split in ['test', 'testing']:
            files = self.test_files
        elif split in ['train', 'training']:
            files = self.train_files
        else:
            raise ValueError("Invalid split {}".format(split))

        return files


class CosegSplit():
    def __init__(self, dataset, split='training'):
        self.cfg = dataset.cfg
        path_list = dataset.get_split_list(split)
        log.info("Found {} pointclouds for {}".format(len(path_list), split))

        self.path_list = path_list
        self.split = split
        self.dataset = dataset


    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        mesh_path = self.path_list[idx]
        mesh_name = mesh_path.split('/')[-1].replace('.obj', '')

        seg_path = join(self.cfg.dataset_path, 'seg', mesh_name + '.eseg')
        label = np.loadtxt(seg_path) - 1

        verts, faces = load_obj(mesh_path)
        faces = faces - 1

        label_v = np.zeros([verts.shape[0]], dtype=np.int64)
        
        edges = construct_edges(faces)
        for ie, e in enumerate(edges):
            label_v[e[0]] = label[ie]
            label_v[e[1]] = label[ie]


        data = {
            'verts': verts, 
            'faces': faces, 
            'label': label_v
        }

        return data

    def get_attr(self, idx):
        mesh_path = self.path_list[idx]
        mesh_name = mesh_path.split('/')[-1].replace('.obj', '')
        class_name = mesh_path.split('/')[-3]

        attr = {
            'name': class_name+mesh_name, 
            'path': mesh_path, 
            'split': self.split
        }
        return attr

