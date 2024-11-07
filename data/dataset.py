import random
import numpy as np
import os
import PIL
from PIL import Image
import cv2
import shutil
import glob

import torch.utils.data.dataset
import torch
from torchvision import transforms


class ImageDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, ):
        super().__init__()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        return sample
