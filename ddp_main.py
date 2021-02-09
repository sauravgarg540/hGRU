import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from data_preprocessing.pre_process import return_image
from data_preprocessing.transform import Resize, ToTorchFormatTensor

import os
import time
import argparse
import configparser
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from model import model
from config import configuration
from Dataset import CustomDataset
from model.hGRU_cell import HgruCell

WORLD_SIZE = torch.cuda.device_count()
config = configparser.ConfigParser()
config.read('config.ini')

def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

def get_dataloader(rank, world_size):
    batch_size = config.getint('train', 'batch_size')
    data_transform = torchvision.transforms.Compose([Resize(), ToTorchFormatTensor()])
    generator = CustomDataset('pf14_train_combined_metadata.txt', transform = data_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(generator, rank=rank, num_replicas=world_size, shuffle=True)
    dataset_loader = torch.utils.data.DataLoader(generator, batch_size=batch_size, num_workers =1, sampler = sampler)
    return dataset_loader

def train(rank, world_size):

    init_process(rank, world_size, "nccl")
    print(
        f"Rank {rank + 1}/{world_size} process initialized."
    )
    
    if rank == 0:  
        get_dataloader(rank, world_size)
        model.hGRU()
    dist.barrier()
    print(f"Rank {rank}/{world_size} training process passed data download barrier.")
    
    net = model.hGRU()
    net.cuda(rank)
    net.train()

    lr = config.getint('train','lr')
    epochs = config.getint('train', 'epochs')

    net = DistributedDataParallel(net, device_ids=[rank])
    dataloader = get_dataloader(rank, world_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr * world_size)
    
        
    for epoch in range(1, epochs+1):
        
        losses = []
     
        for i, (imgs, targets) in enumerate(dataloader):

            optimizer.zero_grad()
            imgs = imgs.cuda(rank)
            targets = targets.cuda(rank)
            output = net(imgs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            if i % 100 == 0:
                 print(
                     f'Finished epoch {epoch}, rank {rank}/{world_size}, batch {i}/{len(dataloader)}. '
                     f'Loss: {curr_loss:.3f}.')
            losses.append(curr_loss)

        # print(
        #     f'Finished epoch {epoch}, rank {rank}/{world_size}. '
        #     f'Avg Loss: {np.mean(losses)}; Median Loss: {np.min(losses)}.\n'
        # )
        
        if rank == 0:
            if not os.path.exists('/spell/checkpoints/'):
                os.mkdir('/spell/checkpoints/')
            torch.save(model.state_dict(), f'/spell/checkpoints/model_{epoch}_.pth')
    torch.save(model.state_dict(), f'/spell/checkpoints/model_final.pth')

def main():
    mp.spawn(train,
        args=(WORLD_SIZE,),
        nprocs=WORLD_SIZE,
        join=True)
    
if __name__ == "__main__":
    main()