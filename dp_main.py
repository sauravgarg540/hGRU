import torch
import torch.nn as nn
from model.hGRU_cell import HgruCell
import torchvision
import numpy as np
from Dataset import CustomDataset

from model import model
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from data_preprocessing.pre_process import return_image
from data_preprocessing.transform import Resize, ToTorchFormatTensor
import os
import time

from statistics import mean
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter
from utils import configuration
writer = SummaryWriter()
torch.manual_seed(32)




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
              
    print(f'Validation-->  Loss: {mean(losses.history):.3f}, Accuracy:{mean(accuracy.history):.3f}, Pecision:{mean(precision.history):.3f} , Recall:{mean(recall.history):.3f}')
    return mean(losses.history), mean(accuracy.history), mean(precision.history), mean(recall.history)

def get_data_loader(config):

    data_transform = torchvision.transforms.Compose([Resize(), ToTorchFormatTensor()])
    
    train_generator = CustomDataset(config['train_dataset'], transform = data_transform)
    val_generator = CustomDataset(config['validation_dataset'], transform = data_transform)
    
    train_loader = torch.utils.data.DataLoader(train_generator, batch_size= config['batch_size'], num_workers = 4)
    val_loader = torch.utils.data.DataLoader(val_generator, batch_size= config['batch_size'], num_workers = 4)
    
    return train_loader, val_loader 


def main(config):

    train_loader, val_loader = get_data_loader(config) 

    if torch.cuda.device_count() > 1:
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        hgru_model = nn.DataParallel(model.hGRU())
    else:
        hgru_model = model.hGRU()

    if torch.cuda.device_count() == 1:
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        hgru_model = model.hGRU()
        hgru_model.cuda()

    exit()


    #checkpnt = torch.load('checkpoints/checkpoint_Jan-31-2021_0010_.pt')
    #hgru_model.load_state_dict(checkpnt['model_state_dict'])
    # print(hgru_model)
    
    # for name, param in hgru_model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.numel())
    # print(sum(p.numel() for p in hgru_model.parameters() if p.requires_grad))
    # exit()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(hgru_model.parameters(), lr=2e-3)
    epochs = 2
    #print_freq = 100
    with torch.autograd.set_detect_anomaly(True):
        validation_loss = AverageMeter()
        validation_accuracy = AverageMeter()
        validation_precision = AverageMeter()
        validation_recall = AverageMeter()
        for epoch in range(epochs):
            hgru_model.train()
            losses = AverageMeter()
            accuracy = AverageMeter()
            precision = AverageMeter()
            recall = AverageMeter()
            
            end = time.perf_counter()
                        
            for i, (imgs, target) in enumerate(train_loader):
                imgs = imgs.cuda()
                target = target.cuda()
                optimizer.zero_grad()
                output  = hgru_model.forward(imgs)
                loss = criterion(output, target)
                
                # add loss to history
                losses.update(loss.item(), imgs.size(0))
                
                loss.backward()
                optimizer.step()
                
                y_true = target.detach().cpu().numpy()
                y_score =  torch.topk(output,1).indices.reshape(output.size(0)).detach().cpu().numpy()
                acc = accuracy_score(y_true, y_score)
                # rec = recall_score(y_true, y_score)
                # prec = precision_score(y_true, y_score)
                accuracy.update(acc, imgs.size(0))#add accuracy to history
                # precision.update(prec, imgs.size(0))
                # recall.update(rec, imgs.size(0))
                # writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
                for n, p in hgru_model.named_parameters():
                    if p.requires_grad:
                        writer.add_scalar(f"{n}/train", p.grad.abs().mean(), epoch * len(train_loader) + i)
                
                if i%1000 ==0:
                    writer.add_scalar(f"Loss_(mean)/train", mean(losses.history),epoch * len(train_loader) + i)
                    writer.add_scalar(f"Accuracy_(mean)/train", mean(accuracy.history), epoch * len(train_loader) + i)
                    print(f'Epoch:{epoch} --> Iteration {i}/{len(train_loader)} Loss: {mean(losses.history):.3f},  Accuracy: {mean(accuracy.history):.3f}') 
                #   print(f'Epoch:{epoch} --> Iteration {i}/{len(train_loader)} Loss: {mean(losses.history):.3f},  Accuracy: {mean(accuracy.history):.3f},  Precision: {mean(precision.history):.3f},'
                #         f'  Recall: {mean(recall.history):.3f}')  
                        
                
            writer.add_scalar("per_epoch_Loss(mean)/train", mean(losses.history), epoch)
            writer.add_scalar("per_epoch_Loss(accuracy)/train", mean(accuracy.history), epoch)   
            # print(f'Epoch:{epoch} --> Loss: {mean(losses.history):.3f},  Accuracy: {mean(accuracy.history):.3f},  Precision: {mean(precision.history):.3f},'
            # f'  Recall: {mean(recall.history):.3f}')
            print(f'Epoch:{epoch} --> Loss: {mean(losses.history):.3f},  Accuracy: {mean(accuracy.history):.3f}')  
            
            
            
            t = time.localtime()
            timestamp = time.strftime('%b-%d-%Y_%H%M', t)
            torch.save({
            'epoch': epoch,
            'model_state_dict': hgru_model.state_dict(),
            }, f"checkpoints/checkpoint_{epoch}_" + timestamp+"_.pt")
            
            np.save(f'dump/accuracy_{epoch}', np.array(accuracy.history))
            np.save(f'dump/loss_{epoch}', np.array(losses.history))
            np.save(f'dump/precision_{epoch}', np.array(precision.history))
            np.save(f'dump/recall_{epoch}', np.array(recall.history))
            #print("validation after {epoch} epochs")
            val_loss, val_accuracy, val_precision, val_recall = evaluate_model(val_loader, hgru_model, criterion)
            validation_loss.update(val_loss,epoch+1)
            validation_accuracy.update(val_accuracy,epoch+1)
            validation_precision.update(val_precision,epoch+1)
            validation_recall.update(val_precision,epoch+1)
    np.save(f'dump/val_accuracy', np.array(validation_accuracy.history))
    np.save(f'dump/val_loss', np.array(validation_loss.history))
    np.save(f'dump/val_precision', np.array(validation_precision.history))
    np.save(f'dump/val_recall', np.array(validation_recall.history))
    
    writer.flush()
    Path = "weights/trained_model"
    torch.save(hgru_model.module, Path)
# # np.array(f_val).dump(open("{}.npy".format(args.name),'w')))
    

if __name__ == "__main__":

    parser = configuration.config()
    config = parser.parse_args()
    main(vars(config))
    
    