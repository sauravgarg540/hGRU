# Feature Extractor in the paper
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


im = Image.open("sample.png")
print(im.size)

resize = torchvision.transforms.Resize(150, 2)
im = resize(im)
print(im.size)

plt.figure()
plt.imshow(im)
plt.show()
