import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math
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

def learnD_Realness(param, D, G, optimizerD, random_sample, Triplet_Loss, x, x1, z1, AR, AF, m):
    device = 'cuda' if param.cuda else 'cpu'
    if m==0:
        z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
        z = z.to(device)

    for p in D.parameters():
        p.requires_grad = True

    for t in range(param.D_updates):
        D.zero_grad()
        optimizerD.zero_grad()

        # gradients are accumulated through subiters
        for _ in range(param.effective_batch_size // param.batch_size):
            if m==0:
                images, _ = random_sample.__next__()
                num_outcomes = Triplet_Loss.atoms
                x.copy_(images)
                del images

            # real images
            if m==0:
                feat_real = D(x)
            else:
                feat_real = D(x1)
            R = feat_real.log_softmax(1).exp()
            # fake images
            if m==0:
                z.normal_(0, 1)
                imgs_fake = G(z)
            else:
                imgs_fake = G(z1)
            feat_fake = D(imgs_fake.detach())
            F = feat_fake.log_softmax(1).exp()

            # compute loss
            lossD_real = Triplet_Loss(AR, R)
            lossD_real.backward()
            lossD_fake = Triplet_Loss(AF, F)
            lossD_fake.backward()
            lossD = lossD_real + lossD_fake

        optimizerD.step()

    return lossD



