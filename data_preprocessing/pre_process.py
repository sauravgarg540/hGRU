"""
Scripts for feature extraction
"""

import torchvision
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt




def return_image():
    im = Image.open("sample.png")
    # print(im.size)
    img = cv2.imread('sample.png')
    # print(img.shape)

    resize = torchvision.transforms.Resize(150, 2)
    im = resize(im)
    # print(im.size)

    # plt.figure()
    # plt.imshow(img)
    # plt.show()

    return img