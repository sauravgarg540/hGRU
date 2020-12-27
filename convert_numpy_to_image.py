from numba import jit
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

@jit
def convert_array_to_image(source, destination):
    for filename in os.listdir(source):
        print(source+'/'+filename)
        data = np.load(source+'/'+filename)
        images = data["images"]
        labels = data["labels"]
        images = images.astype('uint8')
        for i, image in enumerate(images):
            img = Image.fromarray(image, 'L')
            image_directory = destination +'images/'+filename[0:-4]
            if not os.path.exists(image_directory):
                os.makedirs(image_directory)
            img.save(image_directory+'/' +str(i) + '.png')
            label_directory = destination +'labels/'+filename[0:-4]
            if not os.path.exists(label_directory):
                    os.makedirs(label_directory)
                    np.save(label_directory +'/labels.npy', labels)


source = "D:/Study/ULMUniversity/Thesis/dataset/pf6/curv_contour_length_6_full/val"
destination = "D:/Study/ULMUniversity/Thesis/dataset/pf6_with_images/curv_contour_length_6_full/val/"
convert_array_to_image(source, destination)
