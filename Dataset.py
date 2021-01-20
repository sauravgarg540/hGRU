import numpy as np
import os 
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_root, transform = None):
        self.data_root = data_root
        self.image_list = None
        self.transform = transform
        self.parse_txtfile_()
    
    def create_array(self, row):
        return [row[0], row[1]]
    
    def parse_txtfile_(self):
        self.image_list = np.array([self.create_array(row.strip().split(' ')) for row in open(self.data_root)])

    def __len__(self):
        return self.image_list.shape[0]

    def __getitem__(self, index):
        # image = Image.open()
        image = Image.open(self.image_list[index][0])
        label = self.image_list[index][1]
        # print(label)
        image= self.transform(image)
        label = torch.tensor(float(label)).long()
        # print(label)
        # print(image.shape)
        # print(label.shape)
        return image, label



