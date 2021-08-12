import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, in_channels, hidden_dim, n_layer, n_classes):
        super(LSTM, self).__init__()

        self.n_layer = n_layer
        self.latent_dim = 32
        self.hidden_dim = hidden_dim
        self.map = nn.Linear(in_channels, self.latent_dim)
        self.lstm = nn.LSTM(self.latent_dim, hidden_dim, n_layer, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.map(x)
        out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1, :, :]
        y = self.fc(x)
        return y
