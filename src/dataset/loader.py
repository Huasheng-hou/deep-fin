import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset

import os
import pandas as pd
from os.path import join as pjoin


class ACL18_Dataset(Dataset):
    def __init__(self, root, T, split):
        self.root = root
        self.T = T
        self.split = split
        self.data = pd.read_csv(pjoin(self.root, 'T_%d' % self.T, self.split + '.csv'))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        item = self.data.iloc[index]
        x = pd.read_csv(pjoin(self.root, 'T_%d' % self.T, self.split, item['name'] + '.csv'))
        y = item['label']
        x = np.asarray(x)[:, 1:].astype(float)
        return torch.Tensor(x), torch.LongTensor([y])
