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


def get_data_loader(config):

    data_transform = torchvision.transforms.Compose([Resize(config['image_size']), ToTorchFormatTensor()])
    val_generator = CustomDataset(config['validation_dataset'], transform = data_transform)
    val_loader = torch.utils.data.DataLoader(val_generator, batch_size= 32, shuffle=True)
    print('training and validation dataloader created')
    return val_loader 
def plot(images, labels, true):
    f, ax = plt.subplots(4,8)
    count = 0
    for i in range(4):
        for j in range(8):
            # print(i+j)
            # print(images.shape)
            ax[i, j].imshow(images[count,0,:,:], cmap="gray")
            ax[i, j].set_title(f"Predicted:{labels[count]}, Actual:{true[count]}")
            ax[i, j].axis('off')
            count += 1 
    plt.show()

if __name__ == "__main__":
    parser = configuration.config()
    config = parser.parse_args()
    config = vars(config)
    val_loader = get_data_loader(config) 
    net = model.hGRU(config)
    checkpnt = torch.load('weights/pf_14/epoch2.pt')
    net.load_state_dict(checkpnt['model_state_dict'])
    net.cuda()
    net.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets = targets.cuda()
            predict = net.forward(images)
            y_true = targets.detach().cpu().numpy()
            y_score =  torch.topk(predict,1).indices.reshape(predict.size(0)).detach().cpu().numpy()
            plot(images.detach().cpu().numpy(), y_score, y_true)
            
