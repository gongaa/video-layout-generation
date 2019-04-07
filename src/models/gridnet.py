import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import LateralBlock, DownSamplingBlock, UpSamplingBlock

class GridNet(nn.Module):

    def __init__(self, n_channels, filters_out = 3, filters_level = [32, 64, 96],
                 name = 'gridnet'):
        self.n_row = 3
        self.n_col = 6
        self.f_out = filters_out
        self.f_level = filters_level
        self.name = name

        self.lateral_in = LateralBlock(in_ch=n_channels, out_ch=self.f_level[0],
                                       shortcut_conv=True, name='lateral_in')
        self.lateral_out = LateralBlock(in_ch=self.f_level[0], out_ch=filters_out,
                                       shortcut_conv=True, name='lateral_out')
        self.down_0 = [DownSamplingBlock(in_ch=self.f_level[0], out_ch=self.f_level[1], name='down_00')]
        self.down_1 = [DownSamplingBlock(in_ch=self.f_level[1], out_ch=self.f_level[2], name='down_10')]
        self.lateral_0 = []
        self.lateral_1 = []
        self.lateral_2 = []
        self.up_0 = []
        self.up_1 = []
        for i in range(1, int(self.n_col/2)):
            self.lateral_0.append(LateralBlock(in_ch=self.f_level[0], out_ch=self.f_level[0], name=f'lateral_0{i-1}'))
            self.down_0.append(DownSamplingBlock(in_ch=self.f_level[1], out_ch=self.f_level[1], name=f'down_0{i}'))
            self.down_1.append(DownSamplingBlock(in_ch=self.f_level[2], out_ch=self.f_level[2], name=f'down_1{i}'))
            self.lateral_1.append(LateralBlock(in_ch=self.f_level[1], out_ch=self.f_level[1], name=f'lateral_1{i-1}'))
            self.lateral_2.append(LateralBlock(in_ch=self.f_level[2], out_ch=self.f_level[2], name=f'lateral_2{i-1}'))

        for i in range(int(self.n_col/2), self.n_col):
            self.lateral_2.append(LateralBlock(in_ch=self.f_level[2], out_ch=self.f_level[2], name = f'lateral_2{i-1}'))
            self.lateral_1.append(LateralBlock(in_ch=self.f_level[1], out_ch=self.f_level[1], name = f'lateral_1{i-1}'))
            self.lateral_0.append(LateralBlock(in_ch=self.f_level[0], out_ch=self.f_level[0], name = f'lateral_0{i-1}'))
            self.up_1.append(UpSamplingBlock(in_ch=self.f_level[1], out_ch=self.f_level[1], name=f'up_1{i}')) # when use, use self.up_1[i-int(self.n_col/2)]
            self.up_0.append(UpSamplingBlock(in_ch=self.f_level[0], out_ch=self.f_level[0], name=f'up_0{i}')) # when use, use self.up_0[i-int(self.n_col/2)]


    def forward(self, x):
        x_0 = self.lateral_in(x)
        x_1 = self.down_0[0](x_0)
        x_2 = self.down_1[0](x_1)

        for i in range(1, self.n_col):
            if i < self.n_col/2:
                x_0 = self.lateral_0[i-1](x_0)
                x_1 = self.down_0[i](x_0) + self.lateral_1[i-1](x_1)
                x_2 = self.down_1[i](x_1) + self.lateral_2[i-1](x_2)
            else:
                x_2 = self.lateral_2[i-1](x_2)
                x_1 = self.up_1[i-int(self.n_col/2)](x_2) + self.lateral_1[i-1](x_1)
                x_0 = self.up_0[i-int(self.n_col/2)](x_1) + self.lateral_0[i-1](x_0)

        return self.lateral_out(x_0)



    # def __call__(self, x):
    #     with tf.variable_scope(self.name) as vs:
            
    #         x_0 = LateralBlock(out_ch = self.f_level[0], shortcut_conv = True,
    #                            name = 'lateral_in')(x)
    #         x_1 = DownSamplingBlock(out_ch = self.f_level[1], name = 'down_00')(x_0)
    #         x_2 = DownSamplingBlock(out_ch = self.f_level[2], name = 'down_10')(x_1)

    #         for i in range(1, self.n_col):
            
    #             if i < self.n_col/2:
    #                 x_0 = LateralBlock(self.f_level[0], name = f'lateral_0{i-1}')(x_0)
    #                 x_1 = DownSamplingBlock(self.f_level[1], f'down_0{i}')(x_0)\
    #                       + LateralBlock(self.f_level[1], name = f'lateral_1{i-1}')(x_1)
    #                 x_2 = DownSamplingBlock(self.f_level[2], f'down_1{i}')(x_1)\
    #                       + LateralBlock(self.f_level[2], name = f'lateral_2{i-1}')(x_2)
    #             else:
    #                 x_2 = LateralBlock(self.f_level[2], name = f'lateral_2{i-1}')(x_2)
    #                 x_1 = UpSamplingBlock(self.f_level[1], f'up_1{i}')(x_2)\
    #                       + LateralBlock(self.f_level[1], name = f'lateral_1{i-1}')(x_1)
    #                 x_0 = UpSamplingBlock(self.f_level[0], f'up_0{i}')(x_1)\
    #                       + LateralBlock(self.f_level[0], name = f'lateral_0{i-1}')(x_0)

    #         return LateralBlock(self.f_out, True,
    #                             'lateral_out')(x_0)

