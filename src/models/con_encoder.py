import torch
import torch.nn as nn
import torch.nn.functional as F


class ConEncoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(ConEncoder, self).__init__()

    def forward(self, img, seg, mask):
        bs,c,h,w = img.size()
        mask = mask.unsqueeze(1)
        seg_out = seg*mask
        x = torch.cat([img, seg_out], dim=1)

        return x