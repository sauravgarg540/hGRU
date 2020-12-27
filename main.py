import torch
import numpy as np
from config import configuration
from model import model
from data_preprocessing.pre_process import return_image

if __name__ == "__main__":
    # config = configuration('config.ini')
    if torch.cuda.is_available():
        print("CUDA available")
    x = torch.from_numpy(np.zeros((1,1,150,150) ,dtype = np.float32)).cuda()
    cell = model.hGRU(2).cuda()
    # image = return_image()
    cell.forward(x)
    # for name, param in cell.named_parameters():
    #     if param.requires_grad:
    #         print(name)



