import numpy as np
import os 
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root
        self.images = []
        self.labels = []
        for filenames in os.listdir(self.data_root)
            data = np.load(data_root+'/'+filename)
            img = data["images"]
            lbl = data["labels"]
            for image, label in zip(img,lbl):
                img = Image.fromarray(image)
                self.images.append(img)
                self.labels.append(label)

    def __len__(self):
        return len(self.filenames)*1000

    def __getitem__(self, index):
        
        return self.images[index], self.labels[index]

