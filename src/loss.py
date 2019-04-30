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
        self.ssim = SsimLoss()
        self.gd = GradientLoss()

    def forward(self, output, target) -> torch.Tensor:
        return self.vgg(output, target) + self.gd(output, target) + self.ssim(output, target)

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


# class SsimLoss(torch.nn.Module):
#     def __init__(self, window_size=11, size_average=True):
#         super(SsimLoss, self).__init__()
#         self.window_size = window_size
#         self.size_average = size_average
#         self.channel = 1
#         self.window = create_window(window_size, self.channel)

#     def forward(self, img1, img2):
#         (_, channel, _, _) = img1.size()

#         if channel == self.channel and self.window.data.type() == img1.data.type():
#             window = self.window
#         else:
#             window = create_window(self.window_size, channel)

#             if img1.is_cuda:
#                 window = window.cuda(img1.get_device())
#             window = window.type_as(img1)

#             self.window = window
#             self.channel = channel

#         return -_ssim(img1, img2, window, self.window_size, channel, self.size_average)


# def ssim(img1, img2, window_size=11, size_average=True):

#     if len(img1.size()) == 3:
#         img1 = torch.stack([img1], dim=0)
#         img2 = torch.stack([img2], dim=0)

#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)

#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)

#     return _ssim(img1, img2, window, window_size, channel, size_average)


# def gaussian(window_size, sigma):
#     gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
#     return gauss / gauss.sum()


# def create_window(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
#     window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
#     return window


# def _ssim(img1, img2, window, window_size, channel, size_average=True):
#     mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
#     mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
#     sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2

#     ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

#     if size_average:
#         return ssim_map.mean()
#     else:
#         return ssim_map.mean(1).mean(1).mean(1)
