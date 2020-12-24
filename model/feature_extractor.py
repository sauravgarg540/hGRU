# Feature Extractor in the paper

import torch
import torch.nn as nn


class Feature_Extractor(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 25, kernel_size = 7, stride = 1):
        super().__init__()
        self.padding = kernel_size//2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.padding)
        
        # TODO: Initialize weights with gabor filter
    
    def forward(self,x):
        x = self.conv(x)
        return x



(1/2)+1

