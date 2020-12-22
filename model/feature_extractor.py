# Feature Extractor in the paper

import torch
import torch.nn as nn


class Feature_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 25, kernel_size=7, stride=2, padding=1)
    def forward(self,x):
        print("inside feature extractor", x)

