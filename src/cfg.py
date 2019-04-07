import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.CITYSCAPE = edict()

# Initial learning rate
__C.CITYSCAPE.PIXEL_MEANS = np.array([73.15835921, 82.90891754, 72.39239876])
__C.CITYSCAPE.VARS = np.array([[[73.15835921, 82.90891754, 72.39239876]]])



