'''
This is the main file for the project 2's Second method PDE
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d, pad
import os
import matplotlib.pyplot as plt




def pde(img, loc, beta):
    ''' 
    The function to perform the pde update for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. beta: learning rate
    Return:
        img: the updated image
    '''
    img_height, img_width = img.shape
    
    original_pixel = img[loc[0], loc[1]]
    up_pixel = img[max(0,loc[0]-1), loc[1]]
    low_pixel = img[min(img_height-1,loc[0]+1), loc[1]]
    left_pixel = img[loc[0], max(0,loc[1]-1)]
    right_pixel = img[loc[0], min(img_width-1,loc[1]+1)]
    
    x_grad1 = up_pixel - original_pixel
    x_grad2 = original_pixel - low_pixel
    x_2grad = x_grad1 - x_grad2
    y_grad1 = left_pixel - original_pixel
    y_grad2 = original_pixel - right_pixel
    y_2grad = y_grad1 - y_grad2
    
    img[loc[0], loc[1]] = original_pixel + beta*(x_2grad + y_2grad)
    
    return img

def special(img, loc, beta):
    img_height, img_width = img.shape
    
    up_pixel = img[max(0,loc[0]-1), loc[1]]
    low_pixel = img[min(img_height-1,loc[0]+1), loc[1]]
    left_pixel = img[loc[0], max(0,loc[1]-1)]
    right_pixel = img[loc[0], min(img_width-1,loc[1]+1)]
    
    img[loc[0], loc[1]] = up_pixel + low_pixel + left_pixel + right_pixel
    img[loc[0], loc[1]] = img[loc[0], loc[1]]/4
    
    return img

def main():
    # read the distorted image and mask image
    name = "stone"
    size = "big"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)



    beta = 0.3
    img_height, img_width, _ = distort.shape

    sweep = 100
    loss_list = []
    bar = tqdm(total=sweep)
    for s in range(sweep):
        tep_loss_list=[]
        for i in range(img_height):
            for j in range(img_width):
                # only change the channel red
                if mask[i,j,2] == 255:
                    distort[:,:,2] = pde(distort[:,:,2], [i,j], beta)
                    # distort[:,:,2] = special(distort[:,:,2], [i,j], beta)
                    tep_loss_list.append((ori[i,j,2]-distort[i,j,2])**2)

        loss = np.sum(tep_loss_list)/len(tep_loss_list)
        bar.set_description(f"loss: {loss}")
        bar.update(1)
        loss_list.append(loss)

        if s % 10 == 0:
            save_path = f"./result/{name}/{size}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(f"{save_path}/pde_{s}.bmp", distort)
    # plot the loss
    plt.plot(loss_list)
    plt.savefig(f"./result/{name}/{size}/loss.png")



if __name__ == "__main__":
    main()







        

