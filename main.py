import torch
import torch.nn as nn
from model.hGRU_cell import HgruCell
import torchvision
import numpy as np
from Dataset import CustomDataset
from config import configuration
from model import model
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from data_preprocessing.pre_process import return_image
from data_preprocessing.transform import Resize, ToTorchFormatTensor
import os
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def hook_fn(m, i, o):
    print(m)
    print("------------Input Grad------------")

    for grad in i:
        try:
            print(grad.shape)
        except AttributeError: 
            print ("None found for Gradient")

    print("------------Output Grad------------")
    for grad in o:  
        try:
            print(grad.shape)
        except AttributeError: 
            print ("None found for Gradient")
    print("\n")


def tensor_hook(m):
    print(m)

if __name__ == "__main__":
    
    # config = configuration('config.ini')
    data_transform = torchvision.transforms.Compose([Resize(), ToTorchFormatTensor()])
    generator = CustomDataset('pf14_train_combined_metadata.txt', transform = data_transform)
    dataset_loader = torch.utils.data.DataLoader(generator, batch_size=10, shuffle = True)

    if torch.cuda.device_count() > 1:
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        hgru_model = nn.DataParallel(model.hGRU(timesteps = 8))
    elif torch.cuda.is_available():
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        hgru_model = model.hGRU(timesteps = 8)
    hgru_model.cuda()   
    # print(hgru_model)

    # hgru_model.hgru_unit.b_1.register_hook(tensor_hook)
    # hgru_model.b_2.register_hook(tensor_hook)
    # hgru_model.alpha.register_hook(tensor_hook)
    # hgru_model.gamma.register_hook(tensor_hook)
    # hgru_model.kappa.register_hook(tensor_hook)
    # hgru_model.w.register_hook(tensor_hook)
    # hgru_model.mu.register_hook(tensor_hook)
    # hgru_model.n.register_hook(tensor_hook)
   
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(hgru_model.parameters(), lr=0.001)
    epochs = 2
    with torch.autograd.set_detect_anomaly(True): 
        hgru_model.train()
        loss = []
        for epoch in range(epochs):
            for i, (imgs, target) in enumerate(dataset_loader):
                # if i==0:
                #     dummy = torch.randn((1,1,150,150), dtype = torch.float)
                #     writer.add_graph(hgru_model, dummy)
                #     hgru_model.cuda()
                #     print("model added to tensorboard")
                imgs = imgs.cuda()
                target = target.cuda()
                optimizer.zero_grad()
                # to load model on tensorboard

                output  = hgru_model.forward(imgs)
                loss = criterion(output, target)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(hgru_model.parameters(), 1)
                writer.add_scalar("Loss/train", loss, epoch * len(dataset_loader) + i)
                optimizer.step()
                for n, p in hgru_model.named_parameters():
                    writer.add_scalar(f"{n}/train", p.grad.abs().mean(), epoch * len(dataset_loader) + i)
                if i % 5== 0:
                    print(f"Epoch: {epoch}, Iteration: {i}, loss : {loss}")
    # writer.flush()
    Path = "/weights/"
    torch.save(hgru_model.state_dict(), Path)
