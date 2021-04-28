import numpy as np
import torch
import torchvision
import torch.nn as nn
from model.hGRU_cell import HgruCell
import torch.nn.functional as F
from utils import initialization as ini
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import cv2
import numpy as np




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
        self.hgru_unit = HgruCell(writer = self.writer)
        
        # readout stage
        self.conv_readout = nn.Conv2d(25, 2, kernel_size=1)
        self.bn2_1 = nn.BatchNorm2d(25, eps=1e-3, momentum=0.99)
        self.bn2_2 = nn.BatchNorm2d(2, eps=1e-3, momentum=0.99)
        self.maxpool = nn.MaxPool2d(config["image_size"],stride = 1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(2, 2)

        # Weights initialization
        ini.xavier_normal_(self.conv_readout.weight)
        nn.init.zeros_(self.conv_readout.bias)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        

    def forward(self, x, step=None):
        temp = 1
        h_2 = None
        x = self.conv_feature_extractor(x)
        
        if self.writer and step is not None and step%temp == 0:
            self.writer.add_figure("__1_feature_extrctor", gen_plot(x.detach().cpu().numpy()), step)
        x = torch.pow(x, 2) #elementwise multiplication
        
        if self.writer and step is not None and step%temp == 0:
            self.writer.add_figure("__28_element_wise_multiplication", gen_plot(x.detach().cpu().numpy()), step)
        for i in range(self.timesteps):
            h_2 = self.hgru_unit(x, timesteps = i, h_2 = h_2)
            x_ = self.bn2_1(h_2)
            self.writer.add_figure("__32_first_bn", gen_plot(x_.detach().cpu().numpy()), i)
            x_ = self.conv_readout(x_) #[1,2,150,150]

            # add circle at the maximum activation
            x_temp = x_
            x_maxpool1 = self.maxpool(x_)
            x_maxpool1 = self.flat(x_maxpool1)[0] 
            max_1 = (x_temp[0,0] == x_maxpool1[0]).nonzero(as_tuple=True)
            max_2 = (x_temp[0,1] == x_maxpool1[1]).nonzero(as_tuple=True)
            print(max_1)
            print(max_2)
            im = x_.detach().cpu().numpy()[0]
            im1 = cv2.circle(cv2.cvtColor(im[0], cv2.COLOR_GRAY2RGB),(max_1[1],max_1[0]), 5, (255, 255, 0), 2)
            im2 = cv2.circle(cv2.cvtColor(im[1], cv2.COLOR_GRAY2RGB),(max_2[1],max_2[0]), 5, (255, 255, 0), 2)
            im = np.stack((im1, im2))
            self.writer.add_figure("__33_readout", gen_readout(im), i)

            kernel = self.fc.weight.data.reshape(2,2,1,1)
            x_1 = F.conv2d(x_, kernel)
            self.writer.add_figure("__34_heatmap", gen_heatmap(x_1.detach().cpu().numpy()), i)
        if self.writer and step is not None and step%temp == 0:
            self.writer.add_figure("__29_H_2_after_loop", gen_plot(h_2.detach().cpu().numpy()), i)
        x = self.bn2_1(h_2)
        if self.writer and step is not None and step%temp == 0:
            self.writer.add_figure("__30_first_bn", gen_plot(x.detach().cpu().numpy()), i)
        x = self.conv_readout(x) #[1,2,150,150]
        if self.writer and step is not None and step%temp == 0:
            grid_img = torchvision.utils.make_grid(x[0].unsqueeze(1), pad_value = 10)
            self.writer.add_figure("__31_readout", gen_heatmap(x.detach().cpu().numpy()), i)
        # x = torch.max(torch.max(x,2).values,2).values #global maxpooling
        x = self.maxpool(x)
        if self.writer and step is not None and step%temp == 0:
            self.writer.add_scalar("maxpool/1", x[0,0,0,0], step)
            self.writer.add_scalar("maxpool/2", x[0,1,0,0], step)
        x = self.bn2_2(x)
        if self.writer  and step is not None and step%temp == 0:
            self.writer.add_scalar("second_bn/1", x[0,0,0,0], step)
            self.writer.add_scalar("second_bn/2", x[0,1,0,0], step)
        x = self.flat(x)
        x = self.fc(x)
        self.writer.flush()
        return x


def gen_plot(t, sub = True, cmap = 'plasma'):
    """Create a pyplot plot and save to buffer."""
    
    dpi = 10
    _, _, height, width = t.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize= figsize)
    grid = AxesGrid(fig, 111,
                nrows_ncols=(5, 5),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
    count = 0
    #looping through all the kernels in each channel
    for ax in grid:
        if(count==t.shape[1]):
            break
        pcm = ax.imshow(t[0, count],cmap = cmap)
        ax.axis('off')
        count +=1
    cbar = ax.cax.colorbar(pcm)
    cbar = grid.cbar_axes[0].colorbar(pcm)
    # plt.show()
    return fig

def gen_heatmap(t, sub = True, cmap = 'plasma'):
    """Create a pyplot plot and save to buffer."""
    
    dpi = 10
    _, _, height, width = t.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize= figsize)
    grid = AxesGrid(fig, 111,
                nrows_ncols=(1, 2),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
    count = 0
    vmin = np.amin(t[0])
    vmax = np.amax(t[0])
    #looping through all the kernels in each channel
    for ax in grid:
        if(count==t.shape[1]):
            break
        pcm = ax.imshow(t[0, count],vmin = vmin, vmax = vmax, cmap = cmap)
        count +=1
    cbar = ax.cax.colorbar(pcm)
    cbar = grid.cbar_axes[0].colorbar(pcm)
    # plt.show()
    return fig

def gen_readout(t, sub = True, cmap = 'plasma'):
    """Create a pyplot plot and save to buffer."""
    
    dpi = 10
    b, height, width,_ = t.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize= figsize)
    grid = AxesGrid(fig, 111,
                nrows_ncols=(1, 2),
                axes_pad=0.05,
                )
    count = 0
    vmin = np.amin(t)
    vmax = np.amax(t)
    #looping through all the kernels in each channel
    for ax in grid:
        if(count==b):
            break
        pcm = ax.imshow(t[count],vmax = vmax, vmin = vmin, cmap = cmap)
        count +=1
    cbar = ax.cax.colorbar(pcm)
    cbar = grid.cbar_axes[0].colorbar(pcm)
    # plt.show()
    return fig