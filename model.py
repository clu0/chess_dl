from typing import Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn

N = 8
A = 73
N_ACTIONS = N * N * A


def make_train_val_datasets(path: str, train_frac: float = 0.9) -> Tuple["ChessDataset", "ChessDataset"]:
    """
    Read data from path, create a train and val split
    """
    data = np.load(path)
    values = data["values"]
    actions = data["actions"]
    states = data["states"]
    rand_ind = np.random.permutation(len(values))
    n_train = int(train_frac * len(values))
    
    train_dataset = ChessDataset(
        values=values[rand_ind[:n_train]],
        actions=actions[rand_ind[:n_train]],
        states=states[rand_ind[:n_train]],
    )
    val_dataset = ChessDataset(
        values=values[rand_ind[n_train:]],
        actions=actions[rand_ind[n_train:]],
        states=states[rand_ind[n_train:]],
    )
    return train_dataset, val_dataset
    

class ChessDataset(Dataset):
    """
    Dataset class for loading chess data

    Each npz file will contain three arrays:
    states: (n_samples, T * 14 + 7, 8, 8)
    actions: (n_samples, 73, 8, 8)
    values: (n_samples, )
    """
    def __init__(self, values, actions, states) -> None:
        self.values = values
        self.actions = actions
        self.states = states

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
    def __init__(self, in_c, n_c=256, depth=8,  n_hidden=256, width=8):
        super(ChessNet, self).__init__()
        model = [
            nn.Conv2d(in_c, n_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_c),
            nn.ReLU(),
        ]
        for _ in range(depth):
            model += [ResBlock(n_c)]
        self.net = nn.Sequential(*model)
        # global average pooling
        self.policy_features = nn.Conv2d(n_c, N_ACTIONS, kernel_size=3, padding=1)
        # fully connected
        #self.policy_conv = nn.Conv2d(n_c, 10, kernel_size=3, padding=1)
        #self.policy = nn.Linear(10 * N * N, N_ACTIONS)
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
        # global average pooling
        p = self.policy_features(x)
        p = p.mean(dim=(-2, -1))
        p = p.reshape((-1, A, N, N))
        # fully connected
        #p = self.policy_conv(x)
        #p = self.policy(p.reshape((-1, 10 * N * N)))
        #p = p.reshape((-1, A, N, N))
        v = self.value_conv(x)
        v = self.value(v.view(v.size(0),-1))
        return p, v
