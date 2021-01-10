import numpy as np
import torch
import torch.nn as nn
from model.hGRU_cell import HgruCell
from model.feature_extractor import Feature_Extractor
from model.readout import ReadOut



class hGRU(nn.Module):
    
    def __init__(self,timesteps):
        super().__init__()
        self.feature_extractor = Feature_Extractor()
        self.hgu = HgruCell(timesteps = timesteps)
        self.readout = ReadOut()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = np.square(x)
        h_2 = self.hgu(x)
        output = self.readout(h_2)
        return output


# x = torch.from_numpy(np.zeros((1,1,150,150)))
# model = hGRU().cuda()
# model.forward(x)
# p = list(model.parameters())
# for w in p:
#     print(w.shape)
# print(model)
# for name, param in model.state_dict().items():
#     print(name)

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# for param in model.parameters():
#     print(type(param))