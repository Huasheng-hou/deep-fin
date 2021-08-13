import torch
from torch import nn
from torch.utils.data import DataLoader

import torch.nn.functional as f

import os
import sys
import warnings

project = 'ct-seg'
sys.path.append(os.getcwd().split(project)[0] + project)

from model.LSTM import LSTM
from dataset.loader import ACL18_Dataset
from deep2fin import run

warnings.filterwarnings('ignore')

in_channels, hidden_dim, n_layer, n_classes, lr, weight_decay = 6, 16, 2, 2, 1e-1, 0
batch_size = 512

model = LSTM(in_channels, hidden_dim, n_layer, n_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
loss = nn.CrossEntropyLoss()

train_dataset = ACL18_Dataset(root='../data/ACL18/examples', T=10, split='train')
val_dataset = ACL18_Dataset(root='../data/ACL18/examples', T=10, split='val')

t_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
v_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

run(t_loader, v_loader, model, optimizer, loss, n_classes, num_epochs=1000, gpu_idx=0)
