'''
This is the main file for the project 2's first method Gibss Sampler
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
import os
from torch.nn.functional import conv2d, pad
import matplotlib.pyplot as plt


def cal_pot(gradient, norm):
    ''' 
    The function to calculate the potential energy based on the selected norm
    Parameters:
        gradient: the gradient of the image, can be nabla_x or nabla_y, numpy array of size:(img_height,img_width, )
        norm: L1 or L2
    Return:
        A term of the potential energy
    '''
    if norm == "L1":
        return abs(gradient)
    elif norm == "L2":
        return gradient**2 
    else:
        raise ValueError("The norm is not supported!")




def gibbs_sampler(img, loc, energy, beta, norm):
    ''' 
    The function to perform the gibbs sampler for a specific pixel location
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. energy: a scale, refers to the negative exponent
        4. beta: 1/(annealing temperature)
        5. norm: L1 or L2
    Return:
        img: the updated image
    '''
    # get the size of the image
    img_height, img_width = img.shape


    # original pixel value
    original_pixel = img[loc[0], loc[1]]
    
    # for efficiency, only consider the four directions pixels instead of convolving the whole image
    up_pixel = img[max(0,loc[0]-1), loc[1]]
    low_pixel = img[min(img_height-1,loc[0]+1), loc[1]]
    left_pixel = img[loc[0], max(0,loc[1]-1)]
    right_pixel = img[loc[0], min(img_width-1,loc[1]+1)]
    pixel_array = np.array([up_pixel, low_pixel, left_pixel, right_pixel])
    # repeat the array 256 times for broadcasting
    pixel_array = np.repeat(pixel_array, 256).reshape(4,256)
    
    pixel_values = np.arange(256).reshape(1,256)
    
    # calculate the energy for each pixel value
    energy_list = np.sum(cal_pot(pixel_array-pixel_values, norm), axis = 0)
    # print(f"energy_list: {energy_list}")
    # input()
    
    # normalize the energy
    energy_list = energy_list - energy_list.min()
    # energy_list = energy_list / energy_list.sum()

    # calculate the conditional probability
    probs = np.exp(-energy_list * beta)
    # normalize the probs
    probs = probs / probs.sum()

    try:
        # inverse_cdf and updating the img
        img[loc[0], loc[1]] = np.random.choice(np.arange(256), p = probs.squeeze())
    except:
        raise ValueError(f'probs = {probs}')
    return img

def conv(image, filter):
    ''' 
    Computes the filter response on an image.
    Parameters:
        image: numpy array of shape (H, W)
        filter: numpy array of shape, can be [[-1,1]] or [[1],[-1]] or [[1,-1]] or [[-1],[1]] ....
    Return:
        filtered_image: numpy array of shape (H, W)
    '''
    # print(f"image shape: {image.shape}")
    # for the boundery, use periodic boundary condition to pad the image, so that the output size is the same as the input size
    filter = torch.FloatTensor(filter).unsqueeze(0).unsqueeze(0)
    image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)
    pad_size = filter.size(2) // 2
    image = pad(image, (pad_size, pad_size, pad_size, pad_size), mode='circular')
    filtered_image = conv2d(image, filter, padding=0).squeeze().squeeze().numpy()
    

    return filtered_image

def main():
    # read the distorted image and mask image
    name = "room"
    size = "small"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)

    # calculate initial energy
    red_channel = distort[:,:,2]
    energy = 0

    #calculate nabla_x
    filtered_img = conv(red_channel, np.array([[-1,1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))

    # calculate nabla_y
    filtered_img = conv(red_channel, np.array([[-1],[1]]).astype(np.float64))
    energy += np.sum(np.abs(filtered_img), axis = (0,1))



    norm = "L2"
    beta = 0.3
    img_height, img_width, _ = distort.shape

    sweep = 100
    bar= tqdm(total=sweep)
    loss_list = []
    for s in range(sweep):
        loss=0
        tep_loss_list=[]
        for i in range(img_height):
            for j in range(img_width):
                # only change the channel red
                if mask[i,j,2] == 255:
                    distort[:,:,2] = gibbs_sampler(distort[:,:,2], [i,j], energy, beta, norm)
                    tep_loss_list.append((ori[i,j,2]-distort[i,j,2])**2)
        # calculate the loss
        loss = np.sum(tep_loss_list)/len(tep_loss_list)
        loss_list.append(loss)
        bar.set_description(f"loss: {loss}")
        bar.update(1)

        save_path = f"./result/{name}/{norm}/{size}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(f"{save_path}/gibbs_{s}.bmp", distort)
        
    # plot the loss
    plt.plot(loss_list)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(f"Loss of {name} with {norm} norm")
    plt.savefig(f"{save_path}/loss.png")



if __name__ == "__main__":
    main()







        

