import torch
import numpy as np
import sys

class Seq_Dataset(torch.utils.data.Dataset):
    """Dataset for both train and test"""
    def __init__(self, features, targets, seq_lens):
        self.features = features
        self.targets = targets
        self.seq_lens = seq_lens
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        seq_len = self.seq_lens[idx]

        return x, y, seq_len
    
    def get_targets(self):
        return self.targets

class Res_Dataset(torch.utils.data.Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        # self.seq_lens = seq_lens
        
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        # seq_len = self.seq_lens[idx]

        return x, y
    
    def get_targets(self):
        return self.targets