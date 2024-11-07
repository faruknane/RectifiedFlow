import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random


class RectifiedFlowTrajectory:

    def __init__(self):
        pass

    def forward(self, x0, e, t):
        """ 
        x: tensor of shape (B, C, H, W)
        t: tensor of shape (B, )
        """
            
        # Check if t is between 0 and 1
        if not torch.all((0 <= t) & (t <= 1)).item():
            raise ValueError("The value of t must be between 0 and 1 (inclusive).")

        # # random float between 0.99 and 1.01
        # coeff = random.uniform(0.95, 1.05)

        # # e is gaussian noise like x
        # e = torch.randn_like(x0, device=x0.device) * coeff
        
        t = t.view(-1, 1, 1, 1)

        xt = x0 * (1 - t) + t * e
        return xt

    def backward(self, xt, e, t, test_step_size=0.01):
        """
        z: tensor of shape (B, C, H, W)
        e: tensor of shape (B, C, H, W)
        t: tensor of shape (B, )
        """

        # Check if t is between 0 and 1
        if not torch.all((0 <= t) & (t <= 1)).item():
            raise ValueError("The value of t must be between 0 and 1 (inclusive).")
        if not torch.all((0 <= test_step_size) & (test_step_size <= 1)).item():
            raise ValueError("The value of t must be between 0 and 1 (inclusive).")

        t = t.view(-1, 1, 1, 1)
        x0 = (xt - t * e) / (1 - t)

        t2 = t - test_step_size
        x = x0 * (1 - t2) + t2 * e

        return x
    
    def velocity(self, x0, e, t):
        return e - x0




