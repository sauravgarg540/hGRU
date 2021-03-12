import os
import time
import torch
import torch.nn as nn
import torchvision
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
#from utils import nadam
from matplotlib.lines import Line2D
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model import model
from utils import configuration
from Dataset import CustomDataset
from data_preprocessing.pre_process import return_image
from data_preprocessing.transform import Resize, ToTorchFormatTensor
torch.manual_seed(30)

def get_data_loader(config):

    data_transform = torchvision.transforms.Compose([Resize(config['image_size']), ToTorchFormatTensor()])
    val_generator = CustomDataset(config['validation_dataset'], transform = data_transform)
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

# def plot_filters_single_channel(t):
    
#     #kernels depth * number of kernels
#     nplots = t.shape[0]*t.shape[1]
#     ncols = 12
    
#     nrows = 1 + nplots//ncols
#     #convert tensor to numpy image
#     npimg = np.array(t.numpy(), np.float32)
    
#     count = 0
#     fig = plt.figure(figsize=(ncols, nrows))
    
#     #looping through all the kernels in each channel
#     for i in range(t.shape[0]):
#         for j in range(t.shape[1]):
#             count += 1
#             ax1 = fig.add_subplot(nrows, ncols, count)
#             npimg = np.array(t[i, j].numpy(), np.float32)
#             # npimg = (npimg - np.mean(npimg)) / np.std(npimg)
#             # npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
#             ax1.imshow(npimg, cmap="gray")
#             ax1.set_title(str(i) + ',' + str(j))
#             ax1.axis('off')
#             ax1.set_xticklabels([])
#             ax1.set_yticklabels([])
   
#     plt.tight_layout()
#     plt.show()

def plot_filters_single_channel(t):
    
    fig,ax = plt.subplots(nrows=5, ncols=5)
    count = 0
    #looping through all the kernels in each channel
    for i in range(5):
        for j in range(5):
            ax[i,j].imshow(t[count, 0], cmap="gray")
            ax[i, j].axis('off')
            count +=1
    plt.show()


if __name__ == "__main__":
    writer = SummaryWriter()
    # writer = None
    parser = configuration.config()
    config = parser.parse_args()
    config = vars(config)
    val_loader = get_data_loader(config) 
    net = model.hGRU(config, writer)
    checkpnt = torch.load('weights/pf_14/epoch2.pt')
    net.load_state_dict(checkpnt['model_state_dict'])
    # print(net.hgru_unit.n.data)
    # exit()
    # plot_filters_single_channel(net.conv_feature_extractor.weight.data)
    net.cuda()
    net.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            img_grid = torchvision.utils.make_grid(images[0], pad_value = 10, nrow=10)
            writer.add_image('__0_training_images', img_grid, i)
            images = images.cuda()
            targets = targets.cuda()
            predict = net.forward(images, i)
            y_true = targets.detach().cpu().numpy()
            y_score =  torch.topk(predict,1).indices.reshape(predict.size(0)).detach().cpu().numpy()
            print(y_true, y_score)
            # break
            # plot(images.detach().cpu().numpy(), y_score, y_true)
            # im = images.detach().cpu().numpy()
            # plt.figure()
            # plt.imshow(1-im[0,0,:,:], cmap="gray")
            # plt.title(f"Predicted:{y_score}, Actual:{y_true}")
            # plt.axis('off')
            # plt.imshow
    writer.flush()
            
