import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm
import os
from os import makedirs, remove
from os.path import exists, join
import time
import sys
import logging
from datetime import datetime

from torch_dataloader import make_dir
from utils import LogRecord


logging.setLogRecordFactory(LogRecord)
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, model, cfg, chkp_path=None, finetune=False, on_gpu=True):
        self.model = model
        self.cfg = cfg 
        self.epoch = 0
        self.step = 0

        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.adam_lr, 
            weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, cfg.scheduler_gamma)

        model.cfg = cfg
        model.device = self.device
        self.best_acc = 0
        self.best_loss = 1e8

        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        log_file_path = join(cfg.logs_dir, 'log_test_' + timestamp + '.txt')
        log.info("Logging in file : {}".format(log_file_path))
        log.addHandler(logging.FileHandler(log_file_path))
               

    def train(self, train_loader, test_loader=None):
        model = self.model
        cfg = self.cfg
        for epoch in range(cfg.max_epoch):
            model.epoch = epoch
            print("epoch: ", epoch)
            self.metrics = []
            model.train()
            for step, inputs in enumerate(tqdm(train_loader, desc='training')):
                self.optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.get_loss(outputs, inputs)
                loss.backward()
                self.optimizer.step()
                acc = model.get_metric(outputs, inputs)
                self.metrics.append([acc, loss])
            self.scheduler.step()

            model.eval()
            self.test_metrics = []
            with torch.no_grad():
                for step, inputs in enumerate(tqdm(test_loader, desc='test')):
                    outputs = model(inputs)
                    loss = model.get_loss(outputs, inputs)
                    acc = model.get_metric(outputs, inputs)
                    self.test_metrics.append([acc, loss])


            self.compute_result()

            if epoch % cfg.save_ckpt_freq == 0:
                self.save_ckpt(epoch)

        print('Finished Training')
        return

    def save_ckpt(self, epoch):
        path_ckpt = join(self.cfg.logs_dir, 'checkpoint')
        make_dir(path_ckpt)
        torch.save(
            dict(epoch=epoch,
                 model_state_dict=self.model.state_dict(),
                 optimizer_state_dict=self.optimizer.state_dict(),
                 scheduler_state_dict=self.scheduler.state_dict()),
                join(path_ckpt, "ckpt_{}.pth".format(epoch)))
        log.info(f'Epoch {epoch:3d}: save ckpt to {path_ckpt:s}')

    def compute_result(self):
        train_accs = np.array([m[0].cpu().item() for m in self.metrics])
        train_losses = np.array([m[1].cpu().item() for m in self.metrics])
        train_mean_acc = train_accs.mean()
        train_mean_loss = train_losses.mean()

        test_accs = np.array([m[0].cpu().item() for m in self.test_metrics])
        test_losses = np.array([m[1].cpu().item() for m in self.test_metrics])
        test_mean_acc = test_accs.mean()
        test_mean_loss = test_losses.mean()

        self.best_acc = max(self.best_acc, test_mean_acc)
        self.best_loss = min(self.best_loss, test_mean_loss)
        log.info("training -- acc:{:.4f}, loss:{:.4f}".format(train_mean_acc, train_mean_loss))
        log.info("test -- acc:{:.4f}, loss:{:.4f}".format(test_mean_acc, test_mean_loss))
        log.info("best -- acc:{:.4f}, loss:{:.4f}".format(self.best_acc, self.best_loss))
