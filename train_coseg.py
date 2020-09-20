import torch
import os
import time

from torch.utils.data import DataLoader

from laplacian import load_obj
from network import LapCluster
from dataset_coseg import Coseg
from torch_dataloader import TorchDataloader
from trainer import ModelTrainer


class CosegCofig:
    task = 'segmentation'
    dataset_path = '../dataset/coseg_vases'
    logs_dir = '../work_dir/'
    cache_dir = 'cache'
    part_num = 4 
    use_cache = True
    save_ckpt_freq = 20
    pooling = True

    num_eig   = 12
    num_cluster = [24, 16] 
    label_num = 4
    num_points = 480

    lr = 1e-3
    momentum = 0.98
    adam_lr = 0.001
    max_epoch = 800
    weight_decay = 0
    scheduler_gamma = 0.1 ** (1/max_epoch)

    d_feature = 18
    d_in = [64]
    d_b1 = [
        [64, 128],
        [64, 128],
    ]
    d_b2 = [
        [64, 128],
        [64, 128],
    ]
    d_outmlp = [64, 64]
    d_finalmlp = [64, part_num]

    t_augment = {
        'turn_on': False,
        'rotation_method': 'vertical',
        'scale_anisotropic': False,
        'symmetries': False,
        'noise_level': 0,
        'min_s': 0.9,
        'max_s': 1.1,
    }
    
if __name__ == '__main__':

    GPU_ID = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    cfg = CosegCofig()
  
    model = LapCluster(cfg)

    dataset = Coseg(cfg)

    train_split = dataset.get_split('training')


    # Initialize the dataloader
    train_split = TorchDataloader(
        dataset=dataset.get_split('training'),
        use_cache=cfg.use_cache,
        preprocess=model.preprocess,
        transform=model.transform
    )

    train_loader = DataLoader(
        train_split,
        batch_size=8,
        shuffle=True
    )
    test_split = TorchDataloader(
        dataset=dataset.get_split('test'),
        use_cache=cfg.use_cache,
        preprocess=model.preprocess,
        transform=model.transform
    )

    test_loader = DataLoader(
        test_split,
        batch_size=8,
        shuffle=True
    )


    # Define network model
    t1 = time.time()

    # Define a trainer class
    trainer = ModelTrainer(model, cfg)

    # Training
    trainer.train(train_loader, test_loader)

