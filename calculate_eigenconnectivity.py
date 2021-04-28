import torch
import torch.nn as nn
import torchvision
import numpy as np
from model import model
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid

if __name__ == "__main__":
    
    # checkpnt = torch.load('weights/pf_14/epoch2.pt')
    checkpnt = torch.load('checkpoints/pf_14/invert_last_equation/epoch4.pt')
    w_gate = checkpnt['model_state_dict']["hgru_unit.w_gate"]
    w_gate = w_gate.permute(0,2, 3, 1)#change to (I, H, W, O)
    input_channel, height, width, output_channel = w_gate.shape
    rotated_w_gate = np.empty(shape = w_gate.shape)
    
    # rotate filters in the opposite direction
    theta = 0
    w_gate = w_gate.cpu().numpy()
    for i in range(input_channel):
        rotated_w_gate[i] = rotate(w_gate[i], angle = theta, reshape=False)
        theta -=15
    
    rotated_w_gate = np.transpose(rotated_w_gate, (0, 3, 1, 2))#change to (I,O,H,W)


    temp = np.empty((25,25,225))
    for i in range(output_channel):
        for j in range(input_channel):
            temp_ = rotated_w_gate[i, j]
            temp[i,j] = temp_.flatten()
    
    temp = np.concatenate(temp, axis = 0)
    temp = temp.T
    mean = np.mean(temp, axis=0)
    temp = temp - mean
    temp = np.cov(temp)
    eigValues, eigVectors = np.linalg.eig(temp)

    # sort eigen vectors
    idx = eigValues.argsort()[::-1]   
    eigValues = eigValues[idx]
    eigVectors = eigVectors[:,idx]
    # eigValues_sum = eigValues.sum()
    # print(eigValues_sum)
    # print(eigValues)
    # var_percentage = (eigValues/eigValues_sum)*100
    # print(var_percentage)

    # eigVectors_cut = []
    # sum_ = 0

    # for idx, value in enumerate(var_percentage):
    #     if sum_<80:
    #         sum_ += value
    #         eigVectors_cut.append(eigVectors[:,idx])

    # eigVectors_cut = np.array(eigVectors_cut)
    # print(eigVectors_cut.shape)
    # nrows=1
    # ncols=10
    # fig,ax = plt.subplots(nrows=nrows, ncols=ncols, figsize = (38,38))
    # #looping through all the kernels in each channel
    # for i in range(nrows):
    #     for j in range(ncols):
    #         ax[i,j].imshow(eigVectors[:,j].reshape(15,15), cmap="RdBu")
    #         ax[i, j].axis('off')
    # plt.show()

    nrows=2
    ncols=11
    count = 0
    print(eigVectors.shape)
    print(eigVectors[:,0].shape)
    vmin = np.amin(eigVectors[:,0:22])
    vmax = np.max(eigVectors[:,0:22])
    print(vmax, vmin)
    fig,ax = plt.subplots(nrows=nrows, ncols=ncols)
    #looping through all the kernels in each channel
    for i in range(nrows):
        for j in range(ncols):
            test = eigVectors[:,count].reshape(15,15)
            pcm = ax[i,j].imshow(test, cmap="RdBu", vmin = vmin, vmax = vmax)
            ax[i,j].axis('off')
            count+=1
    # divider = make_axes_locatable(ax)
    # colorbar_axes = divider.append_axes("right",
    #                                 size="10%",
    #                                 pad=0.1)
    plt.subplots_adjust(left=0.13,
                    bottom=0.68, 
                    right=0.90, 
                    top=0.88, 
                    wspace=0.05, 
                    hspace=0.00)
    plt.colorbar(pcm, ax = ax[:,:], location = 'right', label = 'PCA score')
    # plt.subplot_tool()
    plt.show()

