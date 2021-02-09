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
            # if (i is 0 and folder is "0"):
            #     combined_data = np.array([[image_dir, str(label[0])]])
            # else:
            #     combined_data = np.append(combined_data,[[image_dir, str(label[0])]], axis = 0)
            File_object.write(image_dir +" "+str(label[0]) +"\n")
        # print("inner loop done")
        # print(folder, type(folder))
        # if(folder == '101'):
            # print(label_source)
    # np.save(label_source+'/train_labels_combines.npy', combined_data)
    File_object.close()
    return
            # return
    #     onlydirs = [os.path.join(d, dd) for dd in os.listdir(d) if os.path.isdir(os.path.join(d, dd))]
    #     if onlydirs:
    #         files += find_files([], onlydirs, contains)

paddel_length = 9
image_source = "D:/Study/ULMUniversity/Thesis/dataset/pf"+str(paddel_length)+"_with_images/curv_contour_length_"+str(paddel_length)+"_full/val/images"
label_source = "D:/Study/ULMUniversity/Thesis/dataset/pf"+str(paddel_length)+"_with_images/curv_contour_length_"+str(paddel_length)+"_full/val/labels"
combine_labels(image_source,label_source)