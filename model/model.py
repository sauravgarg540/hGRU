import numpy as np
import torch
import torch.nn as nn
from model.hGRU_cell import HgruCell
import matplotlib.pyplot as plt

class hGRU(nn.Module):
    
    def __init__(self, config = None):
        
        super().__init__()
        self.timesteps = 8
        # Feature extraction stage
        kernel_size = 7
        self.padding = kernel_size//2
        self.conv_feature_extractor = nn.Conv2d(1, 25, kernel_size= kernel_size, padding=self.padding)
        self.conv_feature_extractor.weight.data = torch.FloatTensor(np.load("gabor_serre.npy"))
        
        # HRGU
        self.hgru_unit = HgruCell()
        
        # readout stage
        self.conv_readout = nn.Conv2d(25, 2, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(25)
        self.bn1 = nn.BatchNorm1d(2)
        self.fc = nn.Linear(2, 2)

        # Weights initialization
        nn.init.xavier_normal_(self.conv_readout.weight)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.zeros_(self.conv_feature_extractor.bias)

    def forward(self, x):

        x = self.conv_feature_extractor(x)
        x = torch.pow(x, 2) #elementwise multiplication
        h_2 = self.hgru_unit(x, timesteps = self.timesteps)
        h_2 = self.bn2(h_2)
        x = self.conv_readout(h_2) #[1,2,150,150]
        x =torch.max(torch.max(x,2).values,2).values #global maxpooling
        x = self.bn1(x)
        x = self.fc(x)
        return x