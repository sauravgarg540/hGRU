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
import time
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

if __name__ == "__main__":
    
    # config = configuration('config.ini')
    data_transform = torchvision.transforms.Compose([Resize(), ToTorchFormatTensor()])
    generator = CustomDataset('pf14_train_combined_metadata.txt', transform = data_transform)
    dataset_loader = torch.utils.data.DataLoader(generator, batch_size= 5, shuffle = True)

    if torch.cuda.device_count() > 1:
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        hgru_model = nn.DataParallel(model.hGRU(timesteps = 8))
    elif torch.cuda.is_available():
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        hgru_model = model.hGRU(timesteps = 8)
    hgru_model.cuda()   

   
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(hgru_model.parameters(), lr=0.001)
    epochs = 2
    with torch.autograd.set_detect_anomaly(True): 
        hgru_model.train()
        loss = []
        for epoch in range(epochs):
            for i, (imgs, target) in enumerate(dataset_loader):
                if i==0:
                    dummy = torch.randn((1,1,150,150), dtype = torch.float).cuda()
                    writer.add_graph(hgru_model, dummy)
               
                imgs = imgs.cuda()
                target = target.cuda()
                optimizer.zero_grad()
                output  = hgru_model.forward(imgs)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if i % 10== 0:
                    writer.add_scalar("Loss/train", loss, epoch * len(dataset_loader) + i)
                    for n, p in hgru_model.named_parameters():
                        writer.add_scalar(f"{n}/train", p.grad.abs().mean(), epoch * len(dataset_loader) + i)
                    print(f"Epoch: {epoch}, Iteration: {i}, loss : {loss}")
                if i% 10== 0:
                    t = time.localtime()
                    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': hgru_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, "checkpoints/checkpoint_" + timestamp+"_.pt")

    writer.flush()
    Path = "/weights/"
    torch.save(hgru_model.state_dict(), Path)
