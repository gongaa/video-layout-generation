import torch
import torch.nn as nn
import torch.nn.functional as F

class LateralBlock(nn.Module):

    def __init__(self, in_ch, out_ch, shortcut_conv = True, name = 'lateral'):
        self.shortcut_conv = shortcut_conv
        self.name = name

        self.conv = nn.Sequential(
            nn.PReLU(),     # can bt called with n_channels
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
       
    def forward(self, x):
        if self.shortcut_conv: # should be called first or final lateral block
            x = self.conv(x) + self.conv2(x)
        else: x = self.conv(x)
        return x
            

class DownSamplingBlock(object):

    def __init__(self, in_ch, out_ch, name = 'down'):
        self.name = name
        self.conv = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)

class UpSamplingBlock(object):

    def __init__(self, in_ch, out_ch, name = 'up'):
        self.name = name
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.PReLU(),
            nn.Conv2d(2*in_ch, out_ch, 3, padding=1),
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
    
    def forward(self, x):
        return self.conv(x)
