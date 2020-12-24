import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

class ConvertArrayToImage:

    def __init__(self, source_path, destination_path):
        self.source_path = source_path
        self.destination_path = destination_path

    def convert_array_to_image(self):
        for filename in os.listdir(self.source_path):
            print(self.source_path+'/'+filename)
            data = np.load(self.source_path+'/'+filename)
            images = data["images"]
            labels = data["labels"]
            images = images.astype('uint8')
            for i,image in enumerate(images):
                img = Image.fromarray(image, 'L')
                image_directory = self.destination_path +'images/'+filename[0:-4]
                if not os.path.exists(image_directory):
                    os.makedirs(image_directory)
                img.save(image_directory+'/' +str(i) + '.png')
            label_directory = self.destination_path +'labels/'
            if not os.path.exists(label_directory):
                    os.makedirs(label_directory)
            np.save(label_directory +'labels.npy', labels)



    def save_image(self, image, destination):
        pass

source = "D:/Study/ULMUniversity/Thesis/dataset/pf6/curv_contour_length_6_full/train"
destination = "D:/Study/ULMUniversity/Thesis/dataset/pf6_with_images/curv_contour_length_6_full/train/"
convert = ConvertArrayToImage(source, destination)
convert.convert_array_to_image()
