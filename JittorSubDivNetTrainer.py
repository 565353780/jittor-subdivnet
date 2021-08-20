#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

from tensorboardX import SummaryWriter

import jittor as jt
import jittor.nn as nn
from jittor.optim import Adam
from jittor.optim import SGD
from jittor.lr_scheduler import MultiStepLR

import numpy as np
from tqdm import tqdm

from subdivnet.dataset import ClassificationDataset
from subdivnet.network import MeshNet
from subdivnet.utils import to_mesh_tensor
from subdivnet.utils import ClassificationMajorityVoting

class SubDivNetTrainer:
    def __init__(self):
        self.reset()

        jt.flags.use_cuda = 1
        return

    def reset(self):
        self.model = None
        self.model_ready = False
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0

        self.input_channels = None
        self.dataroot = None
        self.batch_size = None
        self.n_classes = None
        self.depth = 3
        self.channels = [32, 64, 128, 256]
        self.n_dropout = 2
        self.use_xyz = True
        self.use_normal = True
        self.checkpoint = None
        self.checkpoint_save_name = None

        self.valid_optim_name_list = ['adam', 'sgd']
        self.optim_name = 'adam'
        self.optim = None
        self.optim_ready = False
        self.lr = 1e-3
        self.lr_milestones = None
        self.scheduler = None
        self.weight_decay = 0.0
        self.residual = False
        self.blocks = None
        self.no_center_diff = False
        self.seed = None
        self.n_worker = None
        self.augment_scale = False
        self.augment_orient = False
        self.train_dataset = None
        self.writer = None
        self.train_step = 0
        self.test_best_acc = 0
        self.test_best_vacc = 0
        return

    def resetTimer(self):
        self.time_start = None
        self.total_time_sum = 0
        self.detected_num = 0
        return

    def startTimer(self):
        self.time_start = time.time()
        return

    def endTimer(self, save_time=True):
        time_end = time.time()

        if not save_time:
            return

        if self.time_start is None:
            print("startTimer must run first!")
            return

        if time_end > self.time_start:
            self.total_time_sum += time_end - self.time_start
            self.detected_num += 1
        else:
            print("Time end must > time start!")
        return

    def getAverageTimeMS(self):
        if self.detected_num == 0:
            return -1

        return int(1000.0 * self.total_time_sum / self.detected_num)

    def getAverageFPS(self):
        if self.detected_num == 0:
            return -1

        return int(1.0 * self.detected_num / self.total_time_sum)

    def setCheckPoint(self):
        if self.checkpoint is not None:
            self.model.load(self.checkpoint)
        return

    def setSeed(self, seed):
        self.seed = seed
        if self.seed is not None:
            jt.set_global_seed(self.seed)
        return

    def loadDataset(self, dataroot, batch_size, n_worker, augment_scale, augment_orient):
        self.dataroot = dataroot
        self.batch_size = batch_size
        self.n_worker = n_worker
        self.augment_scale = augment_scale
        self.augment_orient = augment_orient

        augments = []
        if self.augment_scale:
            augments.append('scale')
        if self.augment_orient:
            augments.append('orient')

        self.train_dataset = ClassificationDataset(self.dataroot, batch_size=self.batch_size, 
            shuffle=True, train=True, num_workers=self.n_worker, augment=augments)

        if self.use_xyz:
            self.train_dataset.feats.append('center')
        if self.use_normal:
            self.train_dataset.feats.append('normal')
        return

    def checkOptim(self):
        self.optim_name = optim_name
        self.optim_ready = False

        if self.optim_name not in self.valid_optim_name_list:
            print("optim_name is not valid!")
            return

        if not self.model_ready:
            print("model not ready!")
            return

        if self.optim_name == 'adam':
            self.optim = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optim = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        if self.lr_milestones is not None:
            self.scheduler = MultiStepLR(self.optim, milestones=self.lr_milestones, gamma=0.1)
        else:
            self.scheduler = MultiStepLR(self.optim, milestones=[])

        self.optim_ready = True
        return

    def initEnv(self, dataroot, n_classes, batch_size, n_worker, augment_scale, augment_orient, checkpoint, seed, optim_name):
        self.reset()
        self.n_classes = n_classes
        self.checkpoint = checkpoint
        self.setSeed(seed)
        self.optim_name = optim_name

        self.loadDataset(dataroot, batch_size, n_worker, augment_scale, augment_orient)

        self.input_channels = 7
        if use_xyz:
            self.input_channels += 3
        if use_normal:
            self.input_channels += 3

        if jt.rank == 0:
            self.writer = SummaryWriter("logs/output")
        checkpoint_path = "./checkpoints/output"
        os.makedirs(checkpoint_path, exist_ok=True)
        self.checkpoint_save_name = os.path.join(checkpoint_path, 'subdivnet-latest.pkl')
        return

    def loadModel(self, dataroot,
                  n_classes,
                  batch_size=12,
                  n_worker=4,
                  augment_scale=False,
                  augment_orient=False,
                  checkpoint=None,
                  seed=None,
                  optim_name='adam'):
        self.initEnv(dataroot, n_classes, batch_size, n_worker, augment_scale, augment_orient, checkpoint, seed, optim_name)

        self.model = MeshNet(
            self.input_channels,
            out_channels=self.n_classes, depth=self.depth, 
            layer_channels=self.channels, residual=self.residual, 
            blocks=self.blocks, n_dropout=self.n_dropout, 
            center_diff=not self.no_center_diff)

        self.model_ready = True

        self.checkOptim()
        self.setCheckPoint()

        self.model.train()
        return

    def train_one_epoch(self, epoch_idx):
        n_correct = 0
        n_samples = 0

        disable_tqdm = jt.rank != 0
        for meshes, labels, _ in tqdm(self.train_dataset, desc=f'Train {epoch_idx}', disable=disable_tqdm):

            mesh_tensor = to_mesh_tensor(meshes)
            mesh_labels = jt.int32(labels)

            outputs = self.model(mesh_tensor)
            loss = nn.cross_entropy_loss(outputs, mesh_labels)
            jt.sync_all()
            self.optim.step(loss)
            jt.sync_all()

            preds = np.argmax(outputs.data, axis=1)
            n_correct += np.sum(labels == preds)
            n_samples += outputs.shape[0]

            loss = loss.item()
            if jt.rank == 0:
                self.writer.add_scalar('loss', loss, global_step=self.train_step)

            self.train_step += 1

        jt.sync_all(True)

        if jt.rank == 0:
            acc = n_correct / n_samples
            print('train acc = ', acc)
            self.writer.add_scalar('train-acc', acc, global_step=epoch_idx)
        return

    def train(self, n_epoch):
        for epoch_idx in range(n_epoch):
            self.train_one_epoch(epoch_idx)
            self.scheduler.step()

            jt.sync_all()
            if jt.rank == 0:
                self.model.save(self.checkpoint_save_name)
        return


if __name__ == '__main__':
    dataroot = "/home/chli/3D_FRONT/output/"
    batch_size = 24
    n_classes = 23
    depth = 3
    channels =[32, 64, 128, 256]
    n_dropout = 2
    use_xyz = True
    use_normal = True
    #  checkpoint = "/home/chli/github/jittor-subdivnet/checkpoints/Manifold40.pkl"
    checkpoint = None

    # choices=['adam', 'sgd']
    optim_name = 'adam'
    lr = 1e-3
    lr_milestones = None
    weight_decay = 0.0
    n_epoch = 100
    residual = False
    blocks = None
    no_center_diff = False
    seed = None
    n_worker = 4
    augment_scale = False
    augment_orient = False

    subdivnet_trainer = SubDivNetTrainer()

    subdivnet_trainer.loadModel(dataroot, n_classes, batch_size, n_worker, augment_scale, augment_orient, checkpoint, seed, optim_name)

    subdivnet_trainer.train(n_epoch)

