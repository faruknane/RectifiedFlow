import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# class UniformSampler:
#     def __init__(self, max_time_step, device, include_zero=False, include_last=False):
#         self.max_time_step = max_time_step
#         self.device = device
#         self.include_zero = include_zero
#         self.include_last = include_last

#     def sample(self, batch_size):
#         if self.include_last:
#             y = self.max_time_step+1
#         else:
#             y = self.max_time_step 
        
#         if self.include_zero:
#             x = 0
#         else:
#             x = 1

#         timesteps = torch.randint(x, y, (batch_size, ), dtype=torch.long, device=self.device)

#         return timesteps
    
class LogNormalSampler:

    def __init__(self, m=0, s=1, min_prob = 0.1, resolution=10000):
        self.m = m
        self.s = s
        self.min_prob = min_prob
        self.resolution = resolution

        self.init_distribution()

    def init_distribution(self):
        mu, sigma = self.m, self.s

        t = torch.linspace(0.000001, 0.999999, self.resolution)
        pdf = 1 / (sigma * math.sqrt(2 * math.pi)) * 1 / (t*(1-t)) * torch.exp(-(torch.log(t/(1-t)) - mu)**2 / (2 * sigma**2))

        pdf = pdf / pdf.sum() 
        pdf += self.min_prob / self.resolution
        pdf = pdf / pdf.sum() 
        pdf = pdf * self.resolution

        self.pdf = pdf

    def sample(self, batch_size, device):

        ts = torch.multinomial(self.pdf, batch_size, replacement=True)
        ts = ts / self.resolution

        # to device
        ts = ts.to(device)

        return ts


