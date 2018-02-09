from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import time
import copy
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from caltech256 import Caltech256
from torchvision import models,transforms

## Load Data
#TODO: Data augmentation for training
transform = transforms.Compose(
    [
        transforms.Scale((224,224)),
        transforms.ToTensor(),
    ]
)

data_dir = '/datasets/Caltech256/256_ObjectCategories/'
caltech256_train = Caltech256(data_dir, transform, train=True)
caltech256_test = Caltech256(data_dir, transform, train=False)

train_data = DataLoader(
    dataset = caltech256_train,
    batch_size = 32,
    shuffle = True,
    num_workers = 4
)

test_data = DataLoader(
    dataset = caltech256_test,
    batch_size = 8,
    shuffle = True,
    num_workers = 4
)
