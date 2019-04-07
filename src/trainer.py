import os
import sys
from time import time
import math
import argparse
import itertools
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist 
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets 
from tensorboardX import SummaryWriter 
from utils import AverageMeter

import models

from data import get_dataset
from models.net_utils import *
# from cfg import cfg

def get_model(args):
    # build model
    model = models.__dict__[args.arch](args.embedding_dim)

    # load model
    if args.ckpt is not None:
        args.logger.info('Loading from ckpt %s' % args.ckpt)
        ckpt = torch.load(args.ckpt,
                map_location=torch.device('cpu')) # load to cpu
        if 'model' in ckpt:
            ckpt = ckpt['model']
        model.load_state_dict(ckpt)

    return model


class Trainer:

    def __init__(self, args):
        args.logger.info('Initializing trainer')

        self.model = get_model(args)
        torch.cuda.set_device(args.rank)
        self.model.cuda(args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.rank])
        args.logger.info('Model info\n%s' % str(self.model))

        # if loss is self customed, should write it into module yourself
        self.criterion = nn.CrossEntropyLoss().cuda(args.rank)
        if args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, args.epoch)

        torch.backends.cudnn.benchmark = True
        self.global_step = 0

        if args.ckpt is not None:
            self.load_checkpoint(args.ckpt)

        if args.rank == 0:
            self.writer = SummaryWriter(args.path)

        train_dataset, val_dataset = get_dataset(args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size//args.gpus, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=val_sampler)

        self.args = args

    def set_epoch(self, epoch):
        self.args.logger.info("Start of epoch %d" % (epoch+1))
        self.epoch = epoch + 1
        self.train_loader.sampler.set_epoch(epoch)
        self.val_loader.sampler.set_epoch(epoch)
        if self.args.optimizer == 'sgd':
            self.lr_scheduler.step(epoch)
            if self.args.rank == 0:
                self.writer.add_scalar('other/lr-epoch', self.optimizer.param_groups[0]['lr'], self.epoch)

    def train(self):
        self.args.logger.info('Training started')
        self.model.train()
        end = time()
        for i, (seg, mask, onehot) in enumerate(self.train_loader):
            load_time = time() -end
            end = time()
            # for tensorboard
            self.global_step += 1

            # forward pass
            seg = torch.squeeze(seg, 1)
            seg = seg.cuda(self.args.rank, non_blocking=True)
            mask = mask.cuda(self.args.rank, non_blocking=True)
            onehot = onehot.cuda(self.args.rank, non_blocking=True)
            self.args.logger.debug(seg.shape)
            self.args.logger.debug(mask.shape)
            self.args.logger.debug(onehot.shape)
            _, loss = self.model(seg_gt=seg, mask=mask, onehot=onehot)
            self.args.logger.debug(loss)

            # loss and accuracy
            self.sync([loss])

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            # no need to sync grad because pytorch has already done so
            # self.sync_grads()
            # self.sync_bn_stats()

            # init step in the first train steps
            self.optimizer.step()

            comp_time = time() - end
            end = time()

            # print
            if self.args.rank == 0 and i % self.args.print_freq == 0:
                self.args.logger.info(
                    'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
                    'load [{load_time:.3f}s] comp [{comp_time:.3f}s] '
                    'loss [{loss:.4f}]'.format(
                        epoch=self.epoch, tot_epoch=self.args.epochs,
                        cur_batch=i+1, tot_batch=len(self.train_loader),
                        load_time=load_time, comp_time=comp_time,
                        loss=loss.item()
                    )
                )
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)


    def validate(self):
        self.args.logger.info('Validation started')
        self.model.eval()

        val_loss = AverageMeter()

        with torch.no_grad():
            end = time()
            for i, (seg, mask, onehot) in enumerate(self.val_loader):
                load_time = time()-end
                end = time()

                # forward pass
                seg = torch.squeeze(seg, 1)
                seg = seg.cuda(self.args.rank, non_blocking=True)
                mask = mask.cuda(self.args.rank, non_blocking=True)
                onehot = onehot.cuda(self.args.rank, non_blocking=True)
                _, loss = self.model(seg_gt=seg, mask=mask, onehot=onehot)

                # loss and accuracy
                # onehot.size(0) should be batch size
                size = torch.tensor(float(onehot.size(0))).cuda(self.args.rank) # pylint: disable=not-callable
                loss.mul_(size)
                self.sync([loss], mean=False) # sum
                loss.div_(size)
                val_loss.update(loss.item(), size.item())

                comp_time = time() - end
                end = time()

                # print
                if self.args.rank == 0 and i % self.args.print_freq == 0:
                    self.args.logger.info(
                        'Epoch [{epoch:d}/{tot_epoch:d}][{cur_batch:d}/{tot_batch:d}] '
                        'load [{load_time:.3f}s] comp [{comp_time:.3f}s]'.format(
                            epoch=self.epoch, tot_epoch=self.args.epochs,
                            cur_batch=i+1, tot_batch=len(self.val_loader),
                            load_time=load_time, comp_time=comp_time,
                        )
                    )

        if self.args.rank == 0:
            self.args.logger.info(
                'Epoch [{epoch:d}/{tot_epoch:d}] loss [{loss:.4f}] '.format(
                    epoch=self.epoch, tot_epoch=self.args.epochs,
                    loss=val_loss.avg,
                )
            )
            self.writer.add_scalar('val/loss', val_loss.avg, self.epoch)

        return {'loss': val_loss.avg}

    def sync(self, tensors, mean=True):
        '''Synchronize all tensors given using mean or sum.'''
        for tensor in tensors:
            dist.all_reduce(tensor)
            if mean:
                tensor.div_(self.args.gpus)

    def save_checkpoint(self, metrics):
        self.args.logger.info('Saving checkpoing..')
        prefix = self.args.path + '/checkpoint'
        torch.save({
            'epoch': self.epoch,
            'arch' : self.args.arch,
            'model': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }, '%s/%03d.pth' % (prefix, self.epoch))
        shutil.copy('%s/%03d.pth' % (prefix, self.epoch),
            '%s/latest.pth' % prefix)

    def load_checkpoint(self, ckpt):
        self.args.logger.info('Loading checkpoint %s' % ckpt)
        clpt = torch.load(ckpt)
        assert ckpt['arch'] == self.args.arch, ('Architecture mismatch: ckpt %s, config %s'
                % (ckpt['arch'], self.args.arch))

        self.epoch = ckpt['epoch']
        self.model.load_state_dict(ckpt['optimizer'])
        self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

        self.args.logger.info('Checkpoint loaded')
