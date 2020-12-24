import numpy as np
import torch
import torch.nn as nn
from model.hGRU_cell import HgruCell
from model.feature_extractor import Feature_Extractor



class hGRU(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = Feature_Extractor()
        self.hgu = HgruCell()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.hgu(x)

# x = torch.from_numpy(np.zeros((1,1,150,150)))
# model = hGRU().cuda()
# # model.forward(x)
# # p = list(model.parameters())
# # for w in p:
# #     print(w.shape)
# # print(model)
# # for name, param in model.state_dict().items():
# #     print(name)

# # for name, param in model.named_parameters():
# #     if param.requires_grad:
# #         print(name, param.data)

# for param in model.parameters():
#     print(type(param))