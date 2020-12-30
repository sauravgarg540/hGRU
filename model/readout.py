import numpy as np
import torch
import torch.nn as nn



class ReadOut(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv6 = nn.Conv2d(25, 2, kernel_size=1)
        self.maxpool = nn.MaxPool2d(150, stride=1)
        # self.bn2 = nn.BatchNorm2d(2, eps=1e-03)
        self.fc = nn.Linear(2, 2)
        nn.init.xavier_normal_(self.conv6.weight)
        nn.init.constant_(self.conv6.bias, 0)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.conv6(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x