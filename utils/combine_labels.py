"""Script to combine details aboult all of the dataset in one file for creating dataloader
"""
import numpy as np
import os

def combine_labels(image_source, label_source):
    File_object = open(r"pf9_val_combined_metadata.txt","a+")
    only_folders = [f for f in os.listdir(label_source)]
    for folder in (only_folders):
        print(folder)
        data = np.load(label_source+'/'+folder+'/labels.npy')
        for i, label in enumerate(data):
            image_dir = image_source+'/'+folder+'/'+str(i)+'.png'
            File_object.write(image_dir +" "+str(label[0]) +"\n")
    File_object.close()
    return

paddel_length = 9
image_source = "../dataset/pf"+str(paddel_length)+"_with_images/curv_contour_length_"+str(paddel_length)+"_full/val/images"
label_source = "../dataset/pf"+str(paddel_length)+"_with_images/curv_contour_length_"+str(paddel_length)+"_full/val/labels"
combine_labels(image_source,label_source)