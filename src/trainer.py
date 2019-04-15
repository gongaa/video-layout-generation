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
from torchvision.utils import make_grid
from utils import AverageMeter
from loss import CombinedLoss
import models

from data import get_dataset
from models.net_utils import *
# from cfg import cfg

def get_model(args):
    # build model
    model = models.__dict__[args.arch](n_channels=8)

    # load model
    if getattr(args, 'ckpt', None) is not None:
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
        if not os.path.isdir('../predict'):
            os.makedirs('../predict')
        self.model = get_model(args)
        torch.cuda.set_device(args.rank)
        self.model.cuda(args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.rank])
        self.loss = CombinedLoss()
        self.loss.cuda(args.rank)
        args.logger.info('Model info\n%s' % str(self.model))

        # if loss is self customed, should write it into module yourself

        if args.optimizer == "adamax":
            self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=args.lr)
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        elif args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, args.epoch)

        torch.backends.cudnn.benchmark = True
        self.global_step = 0

        if args.resume is not None:
            self.load_checkpoint(args.resume)

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
        for i, (frame1, seg1, frame2, seg2, frame3) in enumerate(self.train_loader):
            load_time = time() -end
            end = time()
            # for tensorboard
            self.global_step += 1

            # forward pass
            # x = torch.cat([frame1, frame2], dim=1)
            x = torch.cat([seg1, frame1, frame2, seg2], dim=1) # zeroth is batch size
            x = x.cuda(self.args.rank, non_blocking=True)
            frame3 = frame3.cuda(self.args.rank, non_blocking=True)
            # self.args.logger.debug(x.shape)
            img = self.model(x)
            img = 2.5 * F.tanh(img)   # if normalize gt image
            # img = F.sigmoid(img)
            loss = self.loss(output=img, target=frame3)
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

            # save validate result
            # p = torch.cat([frame2.cuda(), frame3, frame2.cuda(), img], dim=1)
            # p = p.cpu().detach().numpy()
            # np.save('../predict/val_'+str(time)+'_'+str(i).zfill(6)+'.npy', p)

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
                self.writer.add_image('train/gt middle', make_grid(frame3, normalize=True), self.global_step)
                self.writer.add_image('train/images', make_grid(img, normalize=True), self.global_step)


    def validate(self):
        self.args.logger.info('Validation started')
        self.model.eval()

        val_loss = AverageMeter()

        with torch.no_grad():
            end = time()
            for i, (frame1, seg1, frame2, seg2, frame3) in enumerate(self.val_loader):
                load_time = time()-end
                end = time()

                # forward pass
                # x = torch.cat([frame1, frame2], dim=1)
                x = torch.cat([seg1, frame1, frame2, seg2], dim=1) # zeroth is batch size, first is channel
                x = x.cuda(self.args.rank, non_blocking=True)
                frame3 = frame3.cuda(self.args.rank, non_blocking=True)
                # self.args.logger.debug(x.shape)
                img = self.model(x=x)
                img = 2.5 * F.tanh(img)   # if normalize gt image
                # img = F.sigmoid(img)
                loss = self.loss(output=img, target=frame3)
                self.args.logger.debug(loss)
                # loss and accuracy
                # img.size(0) should be batch size
                size = torch.tensor(float(img.size(0))).cuda(self.args.rank) # pylint: disable=not-callable
                loss.mul_(size)
                self.sync([loss], mean=False) # sum
                loss.div_(size)
                val_loss.update(loss.item(), size.item())
                # save validate result
                if self.epoch % 2 ==0 and self.args.rank == 0 and i % 100 == 0:
                    p = torch.cat([frame1.cuda(), frame2.cuda(), frame3, img], dim=1)
                    p = p.cpu().detach().numpy()
                    np.save('../predict/val_'+str(end)+'_'+str(i).zfill(6)+'.npy', p)

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
        self.args.logger.info('Saving checkpoint..')
        prefix = '../checkpoint'
        torch.save({
            'epoch': self.epoch,
            'arch': self.args.arch,
            'model': self.model.module.state_dict(), # data parallel
            'optimizer': self.optimizer.state_dict(),
        }, '%s/%03d.pth' % (prefix, self.epoch))
        shutil.copy('%s/%03d.pth' % (prefix, self.epoch),
            '%s/latest.pth' % prefix)

    def load_checkpoint(self, resume):
        self.args.logger.info('Resuming checkpoint %s' % resume)
        ckpt = torch.load(resume)
        assert ckpt['arch'] == self.args.arch, ('Architecture mismatch: ckpt %s, config %s'
            % (ckpt['arch'], self.args.arch))

        self.epoch = ckpt['epoch']
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])

        self.args.logger.info('Checkpoint loaded')