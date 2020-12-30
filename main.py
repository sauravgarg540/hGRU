import torch
import numpy as np
from Dataset import CustomDataset
from config import configuration
from model import model
from data_preprocessing.pre_process import return_image

if __name__ == "__main__":
    # config = configuration('config.ini')

    dataset = CustomDataset("pf6/curv_contour_length_6_full/train")


    if torch.cuda.is_available():
        print("CUDA available")
    # x = torch.from_numpy(np.zeros((1,1,150,150) ,dtype = np.float32)).cuda()
    cell = model.hGRU(2).cuda()
    # cell.forward(x)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cell.parameters(), lr=0.001)
    epochs = 1

    # for epoch in epochs:

    # f_val= []
    # f_training = []
    # train_loss_history = []
    # for epoch in range(1):
    # image = return_image()
    # cell.forward(x)
    # for name, param in cell.named_parameters():
    #     if param.requires_grad:
    #         print(name)
