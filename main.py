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
    cell = model.hGRU().cuda()
    cell.forward(x)

    image = return_image()
    # cell.forward(image, 2)
    # for p in cell.parameters():
    #     print(p.name)



