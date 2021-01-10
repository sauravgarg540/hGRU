import torch
import torchvision
import numpy as np
from Dataset import CustomDataset
from config import configuration
from model import model
from data_preprocessing.pre_process import return_image
from data_preprocessing.transform import Resize, ToTorchFormatTensor

if __name__ == "__main__":
    # config = configuration('config.ini')

    data_transform = torchvision.transforms.Compose([Resize(), ToTorchFormatTensor()])
    generator = CustomDataset('D:/Study/ULMUniversity/Thesis/hGRU/pf_6_train_combined_metadata.txt', transform = data_transform)
    dataset_loader = torch.utils.data.DataLoader(generator, batch_size=32, shuffle = True)
    
    if torch.cuda.is_available():
        print("CUDA available")
    # x = torch.from_numpy(np.zeros((1,1,150,150) ,dtype = np.float32)).cuda()
    cell = model.hGRU(8).cuda()
    # cell.forward(x)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cell.parameters(), lr=0.001)
    epochs = 1
     
    for epoch in range(epochs):
        cell.train()
        for i, (imgs, target) in enumerate(dataset_loader):
            # data_time.update(time.perf_counter() - end)
            
            imgs = imgs.cuda()
            target = target.cuda()
            
            optimizer.zero_grad()
            
            output  = cell.forward(imgs)
            
            loss = criterion(output, target)
            # [prec1] = accuracy(output.data, target, topk=(1,))
            
            # losses.update(loss.data.item(), imgs.size(0))
            # top1.update(prec1.data.item(), imgs.size(0))
            
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            #plot_grad_flow(model.named_parameters())
            optimizer.step()

            # batch_time.update(time.perf_counter() - end)
            
    #         end = time.perf_counter()
    #         if i % (args.print_freq) == 0:
    #             #plot_grad_flow(model.named_parameters())
    #             print('Epoch: [{0}][{1}/{2}]\t lr: {lr:g}\t Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #                    'Prec: {top1.val:.3f} ({precprint:.3f}) ({top1.avg:.3f})\t Loss: {loss.val:.6f} ({lossprint:.6f}) ({loss.avg:.6f})'.format(epoch, i, len(train_loader), batch_time=batch_time,
    #                     data_time=data_time, loss=losses, lossprint= mean(losses.history[-args.print_freq:]), lr=args.lr, top1=top1, precprint= mean(top1.history[-args.print_freq:])))
            
    #     f_training.append(top1.avg)
    #     train_loss_history += losses.history
    #     if (epoch + 1) % 1 == 0 or epoch == args.epochs - 1:
    #         prec = validate(val_loader, model, (epoch + 1) * len(train_loader), criterion)
    #         f_val.append(prec)
    #         is_best = prec > best_prec1
    #         if is_best:
    #             best_prec1 = max(prec, best_prec1)
    #             save_checkpoint({
    #             'epoch': epoch + 1,
    #             'state_dict': model.state_dict(),
    #             'best_prec1': best_prec1,
    #             }, is_best)

    # np.array(f_training).dump(open("{}.npy".format(args.name),'w'))
    # np.array(f_val).dump(open("{}.npy".format(args.name),'w'))


    # f_val= []
    # f_training = []
    # train_loss_history = []
    # for epoch in range(1):
    # image = return_image()
    # cell.forward(x)
    # for name, param in cell.named_parameters():
    #     if param.requires_grad:
    #         print(name)
