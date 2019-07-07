# The VGG loss in this file is copied from
# https://github.com/ekgibbons/pytorch-sepconv/blob/master/python/_support/VggLoss.py
# The SsimLoss loss in this file is copied (with minor modifications) from
# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

from math import exp

import torch
import torch.nn.functional as F
import torchvision
# import src.config as config
from torch import nn
from torch.autograd import Variable


class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, a, b):    # a is output, b is target
        xloss = torch.sum(
            torch.abs(torch.abs(a[:, :, 1:, :] - a[:, :, :-1, :]) - torch.abs(b[:, :, 1:, :] - b[:, :, :-1, :])))
        yloss = torch.sum(
            torch.abs(torch.abs(a[:, :, :, 1:] - a[:, :, :, :-1]) - torch.abs(b[:, :, :, 1:] - b[:, :, :, :-1])))
        return (xloss + yloss) / (a.size()[0] * a.size()[1] * a.size()[2] * a.size()[3])



class VggLoss(nn.Module):
    def __init__(self):
        super(VggLoss, self).__init__()

        model = torchvision.models.vgg19(pretrained=True).cuda()

        self.features = nn.Sequential(
            # stop at relu4_4 (-10)
            *list(model.features.children())[:-10]
        )
        # self.mse = nn.MSELoss(reduction='mean')
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, output, target):
        # add total variation loss
        # reg_loss = 1e-6 * (
        #     torch.sum(torch.abs(output[:, :, :, :-1] - output[:, :, :, 1:])) + 
        #     torch.sum(torch.abs(output[:, :, :-1, :] - output[:, :, 1:, :])))
        outputFeatures = self.features(output)
        targetFeatures = self.features(target)
        
        # loss = torch.norm(outputFeatures - targetFeatures, 2)
        # loss = self.mse(outputFeatures, targetFeatures)
        loss = (outputFeatures - targetFeatures).abs().mean()

        return loss
        # return config.VGG_FACTOR * loss
        # return loss + reg_loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.vgg = VggLoss()
        self.l1 = nn.L1Loss(reduction='mean')
        self.ssim = SsimLoss()
        self.gd = GradientLoss()

    def forward(self, output, target):
        return self.vgg(output, target) + 2*self.l1(output, target) + self.gd(output, target) + self.ssim(output, target)
        # return ( self.vgg(output, target), self.l1(output, target), self.gd(output, target), self.ssim(output, target) )

class SsimLoss(torch.nn.Module):
    def __init__(self):
        super(SsimLoss, self).__init__()
        # self.alpha_recon_image = 0.8
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, 3, 1)
        mu_y = F.avg_pool2d(y, 3, 1)

        sigma_x = F.avg_pool2d(x ** 2, 3, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(y ** 2, 3, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, 3, 1) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1).mean()

    def forward(self, x, y, opt=None):
        sim = 0
        # for ii in range(opt.num_predicted_frames):
        for ii in range(x.size()[1]):
            sim += self.SSIM(x[:, ii, ...], y[:, ii, ...])
        return sim


class PSNR(torch.nn.Module):
    def __init__(self, max_level=1):
        super(PSNR, self).__init__()
        self.max_level = max_level

    def forward(self, pred, gt):
        assert (pred.size() == gt.size())
        _,_,h,w = pred.size()
        psnr = 0
        for i in range(pred.size(0)):
            delta = (pred[i, :, :, :] - gt[i, :, :, :])
            delta = torch.mean(torch.pow(delta, 2))
            psnr += 10 * torch.log10(self.max_level * self.max_level / delta)
        return psnr/pred.size(0)

class IoU(torch.nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, pred, gt):
        assert (pred.size() == gt.size())
        bs,h,w = gt.size()
        true_pixs = (pred == gt).float()
        iou = torch.sum(true_pixs) / (bs*h*w)
        return iou


class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, 
        min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(
                    input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss