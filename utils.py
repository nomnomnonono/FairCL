# Copyright (C) 2019 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Helper functions"""

import os
import random
from glob import glob
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def find_model(path, epoch='latest'):
    if epoch == 'latest':
        files = glob(os.path.join(path, '*.pth'))
        file = sorted(files, key=lambda x: int(x.rsplit('.', 2)[1]))[-1]
    else:
        file = os.path.join(path, 'weights.{:d}.pth'.format(int(epoch)))
    assert os.path.exists(file), 'File not found: ' + file
    print('Find model of {} epoch: {}'.format(epoch, file))
    return file


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
