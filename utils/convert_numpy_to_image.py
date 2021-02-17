import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch

def convert_array_to_image(source, destination):
    for filename in os.listdir(source):
        # print(source+'/'+filename)
        data = np.load(source+'/'+filename)
        images = data["images"]
        labels = data["labels"]
        images = images.astype('uint8')
        for i,image in enumerate(images):
            img = Image.fromarray(image)
            image_directory = destination +'images/'+filename[0:-4]
            if not os.path.exists(image_directory):
                os.makedirs(image_directory)
            img.save(image_directory+'/' +str(i) + '.png')
        label_directory = destination +'labels/'+filename[0:-4]
        if not os.path.exists(label_directory):
                os.makedirs(label_directory)
                np.save(label_directory +'/labels.npy', labels)


source = "../dataset/pf14/curv_contour_length_14_full/val"
destination = "../dataset/pf14_with_images/curv_contour_length_14_full/val/"
convert_array_to_image(source, destination)
