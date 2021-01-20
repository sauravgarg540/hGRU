import numpy as np
import torch
import torch.nn as nn
from model.hGRU_cell import HgruCell
# from model.feature_extractor import Feature_Extractor
# from model.readout import ReadOut
from torchviz import make_dot
import matplotlib.pyplot as plt



class hGRU(nn.Module):
    
    def __init__(self,  in_channels = 1, out_channels = 25, kernel_size = 7, hgru_kernel_size = 15, readout_kernel_size = 1,  stride = 1, timesteps = 8):
        super().__init__()
        self.padding = kernel_size//2
        self.conv_feature_extractor = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.padding)
        self.conv_feature_extractor.weight.data = torch.FloatTensor(np.load("gabor_serre.npy"))
        self.timesteps = timesteps
        
        # readout stage
        self.conv6 = nn.Conv2d(25, 2, kernel_size=1, bias=False)
        self.maxpool = nn.MaxPool2d(150, stride=1)
        self.bn2 = nn.BatchNorm2d(25)
        self.bn1 = nn.BatchNorm1d(2)
        # self.bn2 = nn.BatchNorm2d(2, eps=1e-03)
        self.fc = nn.Linear(2, 2)
        nn.init.xavier_normal_(self.conv6.weight)
        # nn.init.constant_(self.conv6.bias, 0)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        
        nn.init.zeros_(self.conv_feature_extractor.bias)
        # self.feature_extractor = Feature_Extractor()
        # self.hgu = HgruCell(timesteps = timesteps)
        # self.readout = ReadOut()
        # self.timesteps = timesteps
        self.flat = torch.nn.Flatten()

    def forward(self, x):
        x = self.conv_feature_extractor(x)
        x = torch.pow(x, 2) #elementwise multiplication
        self.hgru_unit = HgruCell()
        self.hgru_unit.cuda()
        h_2 = self.hgru_unit(x, timesteps = self.timesteps)
        h_2 = self.bn2(h_2)
        x = self.conv6(h_2)
        x = x.view(x.size(0),x.size(2),x.size(3),x.size(1))
        x = torch.max(torch.max(x,1).values,1).values #global maxpooling
        # print(x.size())
        x = self.bn1(x)
        # x = x.view(x.size(0), -1)
        x = self.flat(x)
        x = self.fc(x)
        return x