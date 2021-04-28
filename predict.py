import os
import time
import io
import torch
import torch.nn as nn
import torchvision
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
#from utils import nadam
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import AxesGrid
import PIL.Image
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import model
from utils import configuration
from Dataset import CustomDataset
from data_preprocessing.pre_process import return_image
from data_preprocessing.transform import Resize, ToTorchFormatTensor
torch.manual_seed(20)

def gen_plot(im):
    """Create a pyplot plot and save to buffer."""
    fig = plt.figure()
    plt.imshow(im[0][0])
    return fig


def get_data_loader(config):

    data_transform = torchvision.transforms.Compose([Resize(config['image_size']), ToTorchFormatTensor()])
    val_generator = CustomDataset("test.txt", transform = data_transform)
    val_loader = torch.utils.data.DataLoader(val_generator, batch_size=1, shuffle=True)
    print('training and validation dataloader created')
    return val_loader 
def plot(images, labels, true):
    f, ax = plt.subplots(4,8)
    count = 0
    for i in range(4):
        for j in range(8):
            ax[i, j].imshow(images[count,0,:,:], cmap="gray")
            ax[i, j].set_title(f"Predicted:{labels[count]}, Actual:{true[count]}")
            ax[i, j].axis('off')
            count += 1 
    plt.show()


def plot_filters_single_channel(t):
    nrows=25
    ncols=25
    fig,ax = plt.subplots(nrows=25, ncols=25, figsize = (38,38))
    count = 0
    #looping through all the kernels in each channel
    grid = AxesGrid(fig, 111,
                nrows_ncols=(25, 25),
                axes_pad=0.05,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
    count = 0
    vmin = np.amin(t)
    vmax = np.amax(t)
    ax = []
    for x in grid:
        ax.append(x)
    count = 0
    #looping through all the kernels in each channel
    for i in range(1, 26):
        for j in range(1, 26):
            pcm = ax[count].imshow(t[i-1, j-1],vmin = vmin, vmax = vmax, cmap = "plasma")
            ax[count].axis('off')
            cbar = ax[count].cax.colorbar(pcm)
            count+=1
    cbar = grid.cbar_axes[0].colorbar(pcm)
    plt.savefig("w_gate.png")
    return


if __name__ == "__main__":
    writer = SummaryWriter()
    # writer = None
    parser = configuration.config()
    config = parser.parse_args()
    config = vars(config)
    val_loader = get_data_loader(config) 
    net = model.hGRU(config, writer)
    checkpnt = torch.load('weights/pf_14/epoch2.pt')
    # checkpnt = torch.load('checkpoints/pf_14/invert_last_equation/epoch4.pt')
    print("weights loaded")
    net.load_state_dict(checkpnt['model_state_dict'])
    # plot_filters_single_channel(net.hgru_unit.w_gate.detach().cpu().numpy())
    # print(net.hgru_unit.alpha.data.T)
    # exit()
    # print(net.fc.weight.data.T)
    # print(net.fc.bias.data.T)
    # print(net.hgru_unit.mix_bias.data.T)
    # print(net.hgru_unit.n.data.T)
    # print(net.hgru_unit.mu.data.T)
    # exit()
    # print(net.hgru_unit.gamma.data.T)
    # print(net.hgru_unit.kappa.data.T)
    # print(net.hgru_unit.omega.data.T)

    # exit()
    # plot_filters_single_channel(net.hgru_unit.w_gate.data)
    # exit()
    net.cuda()
    net.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            writer.add_figure('__0_training_images', gen_plot(images.detach().cpu().numpy()), i)
            images = images.cuda()
            targets = targets.cuda()
            predict = net.forward(images, i)
            y_true = targets.detach().cpu().numpy()
            y_score =  torch.topk(predict,1).indices.reshape(predict.size(0)).detach().cpu().numpy()
            print(y_true, y_score)
            break
            # plot(images.detach().cpu().numpy(), y_score, y_true)
            # im = images.detach().cpu().numpy()
            # plt.figure()
            # plt.imshow(1-im[0,0,:,:], cmap="gray")
            # plt.title(f"Predicted:{y_score}, Actual:{y_true}")
            # plt.axis('off')
            # plt.imshow
    writer.flush()
            