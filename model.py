from typing import Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn


class LargeNPZDataset(Dataset):
    """
    Dataset class for loading chess data
    We preprocess the data from pgn format to npz format. The data gets quite large,
    so the training data will likely be split into a number of files.
    
    Expected data format:
    - data_dir
        - npz files
    
    Each npz file will contain three arrays:
    states: (n_samples, 8 * 14 + 7, 8, 8)
    actions: (n_samples, 73, 8, 8)
    values: (n_samples, )
    """
    def __init__(self, data_dir: str) -> None:
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
        self.global_idx_to_file_idx = []
        global_idx = 0
        for file_path in self.file_list:
            with np.load(file_path) as data:
                local_length = len(data['values'])
            self.global_idx_to_file_idx.extend([(file_path, i) for i in range(local_length)])
            global_idx += local_length

    def __len__(self):
        return len(self.global_idx_to_file_idx)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_path, local_idx = self.global_idx_to_file_idx[idx]
        
        data = np.load(file_path)
        value = data["values"][local_idx]
        action = data["actions"][local_idx]
        state = data["states"][local_idx]

        return state, action, value


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
