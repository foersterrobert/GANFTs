import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from skimage import io

BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PunkDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.transform = transform
        self.image_folder = image_folder
        path, dirs, files = next(os.walk('imgs'))
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, str(idx) + '.png')
        image = io.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return image
