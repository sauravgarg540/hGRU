import numpy as np
import torch
import torchvision
import torch.nn as nn
from model.hGRU_cell import HgruCell
import matplotlib.pyplot as plt
from utils import initialization as ini
import matplotlib.pyplot as plt




class hGRU(nn.Module):
    
    def __init__(self,config = None,writer = None):
        
        super().__init__()
        self.writer = writer
        self.timesteps = 8
        # Feature extraction stage
        kernel_size = 7
        self.padding = kernel_size//2
        self.conv_feature_extractor = nn.Conv2d(1, 25, kernel_size= kernel_size, padding=self.padding)
        self.conv_feature_extractor.weight.data = torch.FloatTensor(np.load("gabor_serre.npy"))
        nn.init.zeros_(self.conv_feature_extractor.bias)
        
        # HRGU
        self.hgru_unit = HgruCell()
        
        # readout stage
        self.conv_readout = nn.Conv2d(25, 2, kernel_size=1)
        self.bn2_1 = nn.BatchNorm2d(25, eps=1e-3, momentum=0.99)
        self.bn2_2 = nn.BatchNorm2d(2, eps=1e-3, momentum=0.99)
        # self.bn1_2 = nn.BatchNorm1d(2, eps=1e-3, momentum=0.99)
        self.maxpool = nn.MaxPool2d(config["image_size"],stride = 1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(2, 2)

        # Weights initialization
        ini.xavier_normal_(self.conv_readout.weight)
        nn.init.zeros_(self.conv_readout.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        

    def forward(self, x, step=None):
        h_2 = None
        x = self.conv_feature_extractor(x)
        if self.writer and step is not None and step%100 == 0:
            grid_img = torchvision.utils.make_grid(x[0].unsqueeze(1), pad_value = 10)
            self.writer.add_image("feature_extrctor", grid_img, step)
        x = torch.pow(x, 2) #elementwise multiplication
        if self.writer and step is not None and step%100 == 0:
            grid_img = torchvision.utils.make_grid(x[0].unsqueeze(1), pad_value = 10)
            self.writer.add_image("element_wise_multiplication", grid_img, step)
        for i in range(self.timesteps):
            h_2 = self.hgru_unit(x, timesteps = i, h_2 = h_2)
        if self.writer and step is not None and step%100 == 0:
            grid_img = torchvision.utils.make_grid(h_2[0].unsqueeze(1), pad_value = 10)
            self.writer.add_image("H_2_images", grid_img, step)
        x = self.bn2_1(h_2)
        if self.writer and step is not None and step%100 == 0:
            grid_img = torchvision.utils.make_grid(x[0].unsqueeze(1), pad_value = 10)
            self.writer.add_image("first_bn", grid_img, step)
        x = self.conv_readout(x) #[1,2,150,150]
        if self.writer and step is not None and step%100 == 0:
            grid_img = torchvision.utils.make_grid(x[0].unsqueeze(1), pad_value = 10)
            self.writer.add_image("readout", grid_img, step)
        # x = torch.max(torch.max(x,2).values,2).values #global maxpooling
        x = self.maxpool(x)
        if self.writer and step is not None and step%100 == 0:
            self.writer.add_scalar("maxpool/1", x[0,0,0,0], step)
            self.writer.add_scalar("maxpool/2", x[0,1,0,0], step)
        x = self.bn2_2(x)
        if self.writer  and step is not None and step%100 == 0:
            self.writer.add_scalar("second_bn/1", x[0,0,0,0], step)
            self.writer.add_scalar("second_bn/2", x[0,1,0,0], step)
        x = self.flat(x)
        x = self.fc(x)
        return x