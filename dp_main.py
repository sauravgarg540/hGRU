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
import matplotlib.pyplot as plt
import os
import time
import configparser
from statistics import mean
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
torch.manual_seed(32)
config = configparser.ConfigParser()
config.read('config.ini')

def evaluate_model(val_loader, model, criterion):
    losses = AverageMeter()
    accuracy = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (imgs, target) in enumerate(val_loader):
        
            target = target.cuda()
            imgs = imgs.cuda()
            
            output = model.forward(imgs)
            
            loss = criterion(output, target)
            losses.update(loss.item(), imgs.size(0))
            
            y_true = target.detach().cpu().numpy()
            y_score =  torch.topk(output,1).indices.reshape(output.size(0)).detach().cpu().numpy()
            acc = accuracy_score(y_true, y_score)
            accuracy.update(acc, imgs.size(0))
            rec = recall_score(y_true, y_score)
            prec = precision_score(y_true, y_score)
            precision.update(prec, imgs.size(0))
            recall.update(rec, imgs.size(0))
            # if i%10 == 0:
            #     print(f'Validation: [{i}/{len(val_loader)}],  Loss: val: {losses.val:.3f} avg:{mean(losses.history):.3f}, Accuracy: val:{accuracy.val:.3f}, avg:{mean(accuracy.history):.3f}')    
    print(f'Validation: [{i}/{len(val_loader)}],  Loss: {mean(losses.history):.3f}, Accuracy:{mean(accuracy.history):.3f}, Pecision:{mean(precision.history):.3f} , Recall:{mean(recall.history):.3f}')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.history = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.history.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    
    # config = configuration('config.ini')
    batch_size = config.getint('train', 'batch_size')

    data_transform = torchvision.transforms.Compose([Resize(), ToTorchFormatTensor()])
    generator = CustomDataset('pf14_train.txt', transform = data_transform)
    dataset_loader = torch.utils.data.DataLoader(generator, batch_size= 10, num_workers = 4, shuffle=True)
    val_generator = CustomDataset('pf_14_1000_val.txt', transform = data_transform)
    val_dataset_loader = torch.utils.data.DataLoader(val_generator, batch_size= 10, num_workers = 4, shuffle=True)

    if torch.cuda.device_count() > 1:
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        hgru_model = nn.DataParallel(model.hGRU())
    elif torch.cuda.is_available():
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        hgru_model = model.hGRU()

    #checkpnt = torch.load('checkpoints/checkpoint_Jan-31-2021_0010_.pt')
    #hgru_model.load_state_dict(checkpnt['model_state_dict'])
    # print(hgru_model)
    # for name, param in hgru_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.numel())
    # print(sum(p.numel() for p in hgru_model.parameters() if p.requires_grad))
    # exit()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(hgru_model.parameters(), lr=1e-4)
    hgru_model = torch.load("weights/trained_model")
    hgru_model.cuda()
    y_true_array = AverageMeter()
    y_score_array = AverageMeter()
    epochs = 10
    print_freq = 100
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(epochs):
            hgru_model.train()
            batch_time = AverageMeter()
            losses = AverageMeter()
            accuracy = AverageMeter()
            precision = AverageMeter()
            recall = AverageMeter()
            val_loss = AverageMeter()
            val_accuracy = AverageMeter()

            end = time.perf_counter()
            
   
            for i, (imgs, target) in enumerate(dataset_loader):
                imgs = imgs.cuda()
                target = target.cuda()
                optimizer.zero_grad()
                output  = hgru_model.forward(imgs)
                loss = criterion(output, target)

                # add loss to history
                losses.update(loss.data.item(), imgs.size(0))

                loss.backward()
                optimizer.step()

                # ad batch time to accuracy
                batch_time.update(time.perf_counter() - end)
                y_true = target.detach().cpu().numpy()
                y_score =  torch.topk(output,1).indices.reshape(output.size(0)).detach().cpu().numpy()
                acc = accuracy_score(y_true, y_score)
                rec = recall_score(y_true, y_score)
                prec = precision_score(y_true, y_score)
                accuracy.update(acc, imgs.size(0))#add accuracy to history
                precision.update(prec, imgs.size(0))
                recall.update(rec, imgs.size(0))
                
                np.save(f'dump/accuracy_{epoch}', np.array(accuracy.history))
                np.save(f'dump/loss_{epoch}', np.array(losses.history))
                np.save(f'dump/precision_{epoch}', np.array(precision.history))
                np.save(f'dump/recall_{epoch}', np.array(recall.history))
                if i%print_freq == 0:

                    print(f'Epoch: {epoch}--{i}/{len(dataset_loader)}],  Loss: val: {losses.val:.3f} avg: {mean(losses.history):.3f},  Batch_time_Average - {batch_time.avg:.3f}'
                    f',  Accuracy: val:{accuracy.val:.3f}, avg:{mean(accuracy.history):.3f},  Precision: val:{precision.val:.3f}, avg:{mean(precision.history):.3f},'
                    f'  Recall: val:{recall.val:.3f}, avg:{mean(recall.history):.3f}')

                if i%1000 == 0:

                    t = time.localtime()
                    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': hgru_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, f"checkpoints/checkpoint_{i}_" + timestamp+"_.pt")
                
            
            print("validation after {epoch} epochs")
            evaluate_model(val_dataset_loader, hgru_model, criterion)
        t = time.localtime()
        timestamp = time.strftime('%b-%d-%Y_%H%M', t)
        torch.save({
        'epoch': epoch,
        'model_state_dict': hgru_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, "checkpoints/checkpoint_" + timestamp+"_.pt")

    Path = "weights/trained_model"
    torch.save(hgru_model, Path)
# np.array(f_val).dump(open("{}.npy".format(args.name),'w')))

