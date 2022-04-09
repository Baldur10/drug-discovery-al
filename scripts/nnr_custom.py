import torch
import numpy as np

from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

# class Torch_Model_BasicRegressor(nn.Module):
#     def __init__(self):
#         super(Torch_Model_BasicRegressor, self).__init__()
#         self.fcs = nn.Sequential(
#             nn.Linear(2048,16384),
#             nn.ReLU(),
#             nn.Dropout(0.15),
#             nn.Linear(16384,256),
#             nn.ReLU(),
#             nn.Dropout(0.15),
#             nn.Linear(256,32),
#             nn.ReLU(),
#             nn.Dropout(0.15),
#             nn.Linear(32,4),
#             nn.ReLU(),
#             nn.Dropout(0.15),
#             nn.Linear(4,1)
#         )
#     def forward(self, x):
#         out = self.fcs(x)
#         return out

# https://arxiv.org/abs/1711.07592
class Torch_Model_BasicRegressor(nn.Module):
    def __init__(self,input_length=2048):
        super(Torch_Model_BasicRegressor, self).__init__()
        self.fcs = nn.Sequential(
            nn.Linear(2048,16384),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(16384,32),
            nn.ReLU(),
            nn.Linear(32,4),
            nn.ReLU(),
            nn.Linear(4,1)
        )

    def forward(self, x):
        out = self.fcs(x)
        return out

class Torch_Model_BasicCNN(nn.Module):
    def __init__(self,):
        super(Torch_Model_BasicCNN, self).__init__()
        self.convs = nn.Sequential(
                                nn.Conv2d(1,32,3),
                                nn.ReLU(),
                                nn.Conv2d(32,64,3),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Dropout(0.25)
        )
        self.fcs = nn.Sequential(
                                nn.Linear(12*12*64,128),
                                nn.ReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128,10),
        )

    def forward(self, x):
        out = x
        out = self.convs(out)
        out = out.view(-1,12*12*64)
        out = self.fcs(out)
        return out