import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import sys
import datetime
import time

def print_now(cmd, file=None):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file is None:
        print('%s %s' % (time_now, cmd))
    else:
        print_str = '%s %s' % (time_now, cmd)
        print(print_str, file=file)
    sys.stdout.flush()

class CategoricalLoss(nn.Module):
    def __init__(self, atoms=50):
        super(CategoricalLoss, self).__init__()
        self.atoms = atoms

    def to(self, device):
        self.device = device

    def forward(self, anchor, feature, skewness=0.0):
        loss = -(anchor * (feature + 1e-16).log()).sum(-1).mean()
        return loss


