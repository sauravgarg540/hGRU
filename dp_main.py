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


torch.manual_seed(42)

def check_path(path):
    if not os.path.exists(path):
        print(f'{path} not found, creating the path')
        os.makedirs(path)


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
        
def evaluate_model(val_loader, model, criterion, config):
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    if config["precision_recall"]:
        train_precision = AverageMeter()
        train_recall = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
         
            targets = targets.cuda()
            images = images.cuda()  
            output = model.forward(images)
            
            loss = criterion(output, targets)
            train_loss.update(loss.item(), images.size(0))
            
            y_true = targets.detach().cpu().numpy()
            y_score =  torch.topk(output,1).indices.reshape(output.size(0)).detach().cpu().numpy()
            acc = accuracy_score(y_true, y_score)
            train_accuracy.update(acc, images.size(0))
            if config["precision_recall"]:
                rec = recall_score(y_true, y_score)
                prec = precision_score(y_true, y_score)
                train_precision.update(prec, images.size(0))
                train_recall.update(rec, images.size(0))
    if config["precision_recall"]:         
        print(f'Validation-->  Loss: {mean(train_loss.history):.3f}, Accuracy:{mean(train_accuracy.history):.3f}, Pecision:{mean(train_precision.history):.3f} , Recall:{mean(train_recall.history):.3f}')
    else:
        print(f'Validation-->  Loss: {mean(train_loss.history):.3f}, Accuracy:{mean(train_accuracy.history):.3f}')
    
    if config["precision_recall"]:
        return mean(train_loss.history), mean(train_accuracy.history), mean(train_precision.history), mean(train_recall.history)
    else:
        return mean(train_loss.history), mean(train_accuracy.history)

def get_data_loader(config):

    check_path(config['checkpoint_path'])
    check_path(config['dump_path'])
    check_path(config['weight_path'])

    data_transform = torchvision.transforms.Compose([Resize(config['image_size']), ToTorchFormatTensor()])
    
    train_generator = CustomDataset(config['train_dataset'], transform = data_transform)
    val_generator = CustomDataset(config['validation_dataset'], transform = data_transform)
    
    train_loader = torch.utils.data.DataLoader(train_generator, batch_size= config['batch_size'], num_workers = 4, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_generator, batch_size= config['batch_size'], num_workers = 4, shuffle = True)
    print('training and validation dataloader created')
    return train_loader, val_loader 


def main(config):
    writer = None

    if config['save_summary']:
        writer = SummaryWriter()
    
    train_loader, val_loader = get_data_loader(config) 

    if torch.cuda.device_count() > 1:
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        net = nn.DataParallel(model.hGRU(config, writer))
    else:
        net = model.hGRU(config, writer)

    if torch.cuda.device_count() >= 1:
        print(f"CUDA available, Model will now train on {torch.cuda.device_count()} GPU's")
        net.cuda()

    if config['load_checkpoint']:
        checkpnt = torch.load('checkpoints/checkpoint_0_Feb-15-2021_0305_.pt')
        print(f"loading checkpoint saved after {checkpnt['epoch']+1} epoch") 
        net.load_state_dict(checkpnt['model_state_dict'])

    print(f'Number of trainable parameters : {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

    if config['load_checkpoint']:
        epochs = checkpnt['epoch']
        epochs = epochs - checkpnt['epoch']+1
    else:
        epochs = 2
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    validation_loss = AverageMeter()
    validation_accuracy = AverageMeter()
    if config['precision_recall']:
        train_precision = AverageMeter()
        train_recall = AverageMeter()          
        validation_precision = AverageMeter()
        validation_recall = AverageMeter()

    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = nadam.Nadam(net.parameters(), lr=config['learning_rate'])
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])
    if config['load_checkpoint']:
        print("Evaluating model after loding checkpoint") 
        evaluate_model(val_loader, net, criterion, config)
    with torch.autograd.set_detect_anomaly(True):
        train_loss.reset()
        train_accuracy.reset()
        validation_loss.reset()
        validation_accuracy.reset()
        if config['precision_recall']:
            train_precision.reset()
            train_recall.reset()          
            validation_precision.reset()
            validation_recall.reset()
        print("starting epochs")
        for epoch in range(epochs):
            net.train()
            for i, (images, targets) in enumerate(train_loader):
                images = images.cuda()
                targets = targets.cuda()
                optimizer.zero_grad()
                output  = net.forward(images,epoch * len(train_loader) + i)
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                
                train_loss.update(loss.item(), images.size(0))
                y_true = targets.detach().cpu().numpy()
                y_score =  torch.topk(output,1).indices.reshape(output.size(0)).detach().cpu().numpy()
                acc = accuracy_score(y_true, y_score)
                train_accuracy.update(acc, images.size(0))

                if config['precision_recall']:
                    rec = recall_score(y_true, y_score)
                    prec = precision_score(y_true, y_score)
                    train_precision.update(prec, images.size(0))
                    train_recall.update(rec, images.size(0))

                if config['save_summary']:
                    if i%100 ==0:
                        img_grid = torchvision.utils.make_grid(images[0], pad_value = 10, nrow=10)
                        writer.add_image('training_images', img_grid, epoch * len(train_loader) + i)
                    writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
                    for n, p in net.named_parameters():
                        if p.requires_grad:
                            writer.add_scalar(f"{n}/train", p.grad.abs().mean(), epoch * len(train_loader) + i)
                
                if i%config['print_frequency'] ==0:
                    if config['save_summary']:
                        writer.add_scalar(f"Loss_(mean)/train", mean(train_loss.history),epoch * len(train_loader) + i)
                        writer.add_scalar(f"Accuracy_(mean)/train", mean(train_accuracy.history), epoch * len(train_loader) + i)
                    if config['precision_recall']:
                        print(f'Epoch:{epoch} --> Iteration {i}/{len(train_loader)} Loss: {mean(train_loss.history):.3f},  Accuracy: {mean(train_accuracy.history):.3f}, '
                            f'Precision: {mean(train_precision.history):.3f}, Recall: {mean(train_recall.history):.3f}') 
                        
                    else: 
                        print(f'Epoch:{epoch} --> Iteration {i}/{len(train_loader)} Loss: {mean(train_loss.history):.3f},  Accuracy: {mean(train_accuracy.history):.3f}') 
            if config['save_summary']:   
                writer.add_scalar("per_epoch_Loss(mean)/train", mean(train_loss.history), epoch)
                writer.add_scalar("per_epoch_Loss(train_accuracy)/train", mean(train_accuracy.history), epoch) 
                if config['precision_recall']:  
                    print(f'Epoch:{epoch} --> Loss: {mean(train_loss.history):.3f},  Accuracy: {mean(train_accuracy.history):.3f}, '
                    f'Precision: {mean(train_precision.history):.3f}, Recall: {mean(train_recall.history):.3f}')
                else:
                    print(f'Epoch:{epoch} --> Loss: {mean(train_loss.history):.3f},  Accuracy: {mean(train_accuracy.history):.3f}')  
            
            
            
            t = time.localtime()
            timestamp = time.strftime('%b-%d-%Y_%H%M', t)
            torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            }, f"checkpoints/checkpoint_{epoch}_" + timestamp+"_.pt")

            if(config['save_dump']):
                np.save(f'dump/accuracy_{epoch}', np.array(train_accuracy.history))
                np.save(f'dump/loss_{epoch}', np.array(train_loss.history))
                if config['precision_recall']: 
                    np.save(f'dump/precision_{epoch}', np.array(train_precision.history))
                    np.save(f'dump/recall_{epoch}', np.array(train_recall.history))
                
            
            if config["precision_recall"]:
                val_loss, val_accuracy, val_precision, val_recall = evaluate_model(val_loader, net, criterion, config)
            else:
                val_loss, val_accuracy = evaluate_model(val_loader, net, criterion, config)

            validation_loss.update(val_loss,epoch+1)
            validation_accuracy.update(val_accuracy,epoch+1)
            if config['precision_recall']: 
                validation_precision.update(val_precision,epoch+1)
                validation_recall.update(val_precision,epoch+1)

    if(config['save_dump']):
        dump_path = config['dump_path']
        np.save(f'{dump_path}/val_accuracy', np.array(validation_accuracy.history))
        np.save(f'{dump_path}/val_loss', np.array(validation_loss.history))
        if config['precision_recall']: 
            np.save(f'{dump_path}/val_precision', np.array(validation_precision.history))
            np.save(f'{dump_path}/val_recall', np.array(validation_recall.history))
    
    writer.flush()

if __name__ == "__main__":

    parser = configuration.config()
    config = parser.parse_args()
    print(vars(config))
    main(vars(config))
    
    