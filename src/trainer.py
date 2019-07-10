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
from utils import AverageMeter, adjust_learning_rate
from loss import CombinedLoss, PSNR, CrossEntropy, OhemCrossEntropy
import models
import cv2
import random
from data import get_dataset
from models.net_utils import *
from config import config, update_config

from models.seg_hrnet import get_seg_model

color_map = [
[128, 64, 128] ,   # road
[244, 35, 232]  ,  # sidewald
[70, 70, 70] ,  # building
[102, 102, 156] , # wall
[190, 153, 153] , # fence
[153, 153, 153] , # pole
[250, 170, 30] , # traffic light
[220, 220, 0] ,  # traffic sign
[107, 142, 35] , # vegetation
[152, 251, 152] , # terrain
[70, 130, 180]  , # sky
[220, 20, 60] , # person
[255, 0, 0]  , # rider
[0, 0, 142]   , # car
[0, 0, 70]  ,  # truck
[0, 60, 100] ,  # bus
[0, 80, 100] ,  # on rails / train
[0, 0, 230]  , # motorcycle
[119, 11, 32] , # bicycle
[0, 0, 0]   # None
]

def get_model(args):
    # build model
    model = models.__dict__[args.arch](n_channels=8, args=args)

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

        ################### HRNet #############################
        update_config(config, args)
        args.logger.info('config info')
        args.logger.info(config)
        self.model = get_seg_model(config)
        torch.cuda.set_device(args.rank)
        self.model.cuda(args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                device_ids=[args.rank])
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507, 0]).cuda()
        self.seg_loss = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL,
                                     thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP,
                                     weight=self.class_weights)
        self.seg_loss.cuda(args.rank)
        self.img_loss = CombinedLoss()
        self.img_loss.cuda(args.rank)
        self.optimizer = torch.optim.SGD([{'params': filter(lambda p: p.requires_grad, self.model.parameters()),
                'lr': config.TRAIN.LR}],lr=config.TRAIN.LR,momentum=config.TRAIN.MOMENTUM,
                weight_decay=config.TRAIN.WD,nesterov=config.TRAIN.NESTEROV,)
        #####################################################
        self.mean_arr = torch.tensor([-0.03,-0.088,-0.188])[None,:,None,None].cuda(args.rank)
        self.std_arr = torch.tensor([0.458,0.448,0.450])[None,:,None,None].cuda(args.rank)
        # self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
        # self.loss = CombinedLoss()
        # self.psnr = PSNR()
        # self.cross_entropy_loss.cuda(args.rank)
        # self.loss.cuda(args.rank)
        # self.psnr.cuda(args.rank)
        # args.logger.info('Model info\n%s' % str(self.model))

        # if loss is self customed, should write it into module yourself

        # if args.optimizer == "adamax":
        #     self.optimizer = torch.optim.Adamax(self.model.parameters(), lr=args.lr)
        # elif args.optimizer == "adam":
        #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        # elif args.optimizer == "sgd":
        #     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9)
        #     self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #             self.optimizer, args.epoch)

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
        self.epoch = 0

        self.epoch_iters = np.int(self.train_loader.__len__() / args.batch_size) 
        self.num_iters = args.epochs * self.epoch_iters

        self.args.logger.debug('Finish init trainer')

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
        for i, (frame1, seg1, frame2, seg2, frame3, seg3) in enumerate(self.train_loader):
            load_time = time() -end
            end = time()
            # for tensorboard
            self.global_step += 1
            cur_iters = self.epoch * self.epoch_iters

            # forward pass
            x = torch.cat([seg1, frame1, frame2, seg2], dim=1) # zeroth is batch size
            x = x.cuda(self.args.rank, non_blocking=True)
            frame3 = frame3.cuda(self.args.rank, non_blocking=True)
            seg3 = seg3.cuda(self.args.rank, non_blocking=True)

            # random flipping
            if random.random() < 0.5:
                with torch.no_grad():
                    x = torch.flip(x, [3])
                    frame3 = torch.flip(frame3, [3])
                    seg3 = torch.flip(seg3, [2]) # N*H*W

            # self.args.logger.debug(x.shape)
            seg, img = self.model(x)
            # normalize img from range [-1,1] to range [(0-0.485)/0.229, (1-0.485)/0.229] for the first channel
            img  = nn.Tanh()(img)
            img = (img - self.mean_arr) / self.std_arr
            
            # vgg_loss, l1_loss, gd_loss, ssim_loss = self.loss(output=img, target=frame3)
            # img_loss = (vgg_loss+2*l1_loss+gd_loss+ssim_loss) * 20
            img_loss = self.loss(output=img, target=frame3) * 20
            seg_loss = self.cross_entropy_loss(input=seg, target=seg3) * 10
            loss = img_loss + seg_loss
            # loss_dict = {'vgg': vgg_loss.item(), 'l1': l1_loss.item(), 'gd': gd_loss.item(), 'ssim': ssim_loss.item()}
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

            lr = adjust_learning_rate(self.optimizer, config.TRAIN.LR, self.num_iters, i + cur_iters)
            
            # seg = torch.argmax(seg, dim=1)
            seg = self.vis_seg_mask(seg, 20, argmax=True)
            seg_gt = self.vis_seg_mask(seg3, 20, argmax=False)

            # inverse normalize of mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # both apply to frame3 and img
            # to be written in jupyter notebook while viewing validation result
            
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
                self.writer.add_scalar('train/img loss', img_loss.item(), self.global_step)
                # self.writer.add_scalar('train/img loss detail', loss_dict, self.global_step)
                self.writer.add_scalar('train/seg loss', seg_loss.item(), self.global_step)
                self.writer.add_image('train/img gt', make_grid(frame3, normalize=True), self.global_step)
                self.writer.add_image('train/img', make_grid(img, normalize=True), self.global_step)
                self.writer.add_image('train/seg gt', make_grid(seg_gt, normalize=True), self.global_step)
                self.writer.add_image('train/seg', make_grid(seg, normalize=True), self.global_step)


    def validate(self):
        self.args.logger.info('Validation started')
        self.model.eval()

        val_loss = AverageMeter()

        with torch.no_grad():
            end = time()
            for i, (frame1, seg1, frame2, seg2, frame3, seg3) in enumerate(self.val_loader):
                load_time = time()-end
                end = time()

                # forward pass
                x = torch.cat([seg1, frame1, frame2, seg2], dim=1) # zeroth is batch size, first is channel
                x = x.cuda(self.args.rank, non_blocking=True)
                frame3 = frame3.cuda(self.args.rank, non_blocking=True)
                seg3 = seg3.cuda(self.args.rank, non_blocking=True)
                seg, img = self.model(x)
                # normalize img from range [-1,1] to range [(0-0.485)/0.229, (1-0.485)/0.229] for the first channel
                img = nn.Tanh()(img)
                img = (img - self.mean_arr) / self.std_arr

                # vgg_loss, l1_loss, gd_loss, ssim_loss = self.loss(output=img, target=frame3)
                # img_loss = (vgg_loss+2*l1_loss+gd_loss+ssim_loss) * 20
                img_loss = self.loss(output=img, target=frame3) * 20
                # psnr_loss = self.psnr(pred=img, gt=frame3)
                seg_loss = self.cross_entropy_loss(input=seg, target=seg3) * 10
                loss = img_loss + seg_loss
                self.args.logger.debug(loss)
                # loss_dict = {'vgg': vgg_loss.item(), 'l1': l1_loss.item(), 'gd': gd_loss.item(), 'ssim': ssim_loss.item(), 'psnr': psnr_loss.item()}

                # loss and accuracy
                # img.size(0) should be batch size
                size = torch.tensor(float(img.size(0))).cuda(self.args.rank) # pylint: disable=not-callable
                loss.mul_(size)
                self.sync([loss, size], mean=False) # sum
                loss.div_(size)
                val_loss.update(loss.item(), size.item())


                seg = torch.argmax(seg, dim=1).unsqueeze_(1).float() # from NCHW to N1HW
                seg3 = seg3.unsqueeze_(1).float() # from [N,H,W] to [N,1,H,W]

                # save validate result
                if self.epoch % 1 == 0 and self.args.rank == 0 and i % 100 == 0:
                    p = torch.cat([frame1.cuda(), frame2.cuda(), frame3, img, seg1.cuda(), seg2.cuda(), seg3, seg], dim=1)
                    p = p.cpu().detach().numpy()
                    np.save('../predict/val_'+str(end)+'_'+str(i).zfill(6)+'.npy', p)

                if self.epoch > 10 and self.epoch % 5 == 0 and self.args.rank == 0 and i % 10 == 0:
                    self.generate_sequence(frame1, frame2, seg1, seg2, self.epoch)
                
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
            # self.writer.add_scalar('loss detail', loss_dict, self.epoch)

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

    def vis_seg_mask(self, seg, n_classes, argmax=False):
        '''
        mask (bs, c,h,w) into normed rgb (bs, 3,h,w)
        all tensors
        '''
        global color_map
        if argmax:
            id_seg = torch.argmax(seg, dim=1)
        else: id_seg = seg
        color_mapp = torch.tensor(color_map)
        rgb_seg = color_mapp[id_seg].permute(0,3,1,2).contiguous().float()
        return rgb_seg/255

    def eval_generate_sequence(self, img1, img2, seg1, seg2):
        seg1 = cv2.imread(seg1, cv2.IMREAD_GRAYSCALE)
        seg2 = cv2.imread(seg2, cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(img1)
        img2 = cv2.imread(img2)
        if seg1 is None or seg2 is None or img1 is None or img2 is None:
            self.args.logger.debug('path name not exists')
            return
        seg1 = cv2.resize(seg1, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        seg2 = cv2.resize(seg2, dsize=(256,256), interpolation=cv2.INTER_NEAREST)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) 
        to_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
        ])
        img1, img2 = to_normalize(img1).unsqueeze_(0), to_normalize(img2).unsqueeze_(0)
        seg1, seg2 = torch.from_numpy(seg1), torch.from_numpy(seg2)
        seg1, seg2 = seg1.float().unsqueeze_(0).unsqueeze_(0), seg2.float().unsqueeze_(0).unsqueeze_(0)
        self.args.logger.debug(seg1.shape)
        self.args.logger.debug(img1.shape)
        self.generate_sequence(img1, img2, seg1, seg2)


    def generate_sequence(self, img1, img2, seg1, seg2, epoch):
        img, seg = [], []
        img.append(img1.cuda(self.args.rank, non_blocking=True))
        img.append(img2.cuda(self.args.rank, non_blocking=True))
        seg.append(seg1.cuda(self.args.rank, non_blocking=True))
        seg.append(seg2.cuda(self.args.rank, non_blocking=True))
        with torch.no_grad():
            for i in range(8):
                x = torch.cat([seg[-2], img[-2], img[-1], seg[-1]], dim=1) # zeroth is batch size, first is channel
                x = x.cuda(self.args.rank, non_blocking=True)
                # self.args.logger.debug(x.shape)
                seg_next, img_next = self.model(x)
                img_next = F.tanh(img_next)
                img_next = (img_next - self.mean_arr) / self.std_arr
                seg_next = torch.argmax(seg_next, dim=1).unsqueeze_(1).float() # from NCHW to N1HW
                img.append(img_next)
                seg.append(seg_next)
            p = torch.cat(img, dim=1)
            q = torch.cat(seg, dim=1)
            p = p.cpu().detach().numpy()
            q = q.cpu().detach().numpy()
            t = time()
            np.save('../predict/val_epoch'+str(epoch)+'_'+str(t)+'_'+'img'+'.npy', p)
            np.save('../predict/val_epoch'+str(epoch)+'_'+str(t)+'_'+'seg'+'.npy', q)







