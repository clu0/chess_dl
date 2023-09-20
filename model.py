from typing import Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn


class ChessDataset(Dataset):
    """
    Dataset class for loading chess data

    Each npz file will contain three arrays:
    states: (n_samples, 8 * 14 + 7, 8, 8)
    actions: (n_samples, 73, 8, 8)
    values: (n_samples, )
    """
    def __init__(self, path: str) -> None:
        data = np.load(path)
        self.values = data["values"]
        self.actions = data["actions"]
        self.states = data["states"]

    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[idx], self.actions[idx], self.values[idx]


class ResBlock(nn.Module):
    def __init__(self, c):
        super(ResBlock, self).__init__()
        res = [
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
        ]
        self.res = nn.Sequential(*res)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = x + self.res(x)
        out = self.relu(x)
        return out

class ChessNet(nn.Module):
    def __init__(self, in_c, n_c=256, depth=8, n_a=73, n_hidden=256, width=8):
        super(ChessNet, self).__init__()
        model = [
            nn.Conv2d(in_c, n_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_c),
            nn.ReLU(),
        ]
        for _ in range(depth):
            model += [ResBlock(n_c)]
        self.net = nn.Sequential(*model)
        self.policy = nn.Sequential(
            nn.Conv2d(n_c, n_a, kernel_size=1),
            nn.ReLU()
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(n_c, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(width**2, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.net(x)
        p = self.policy(x)
        v = self.value_conv(x)
        v = self.value(v.view(v.size(0),-1))
        return p, v
