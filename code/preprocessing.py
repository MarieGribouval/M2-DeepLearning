# -*- coding: utf-8 -*-
'''
# Created on Thu Nov 14 10:34:58 2019
# @title : Deep Bees - Data Extraction and Cleansing
# @author: Marie Gribouval, Léo Boule and Leshanshui YANG
'''


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import imageio
from PIL import Image
from PIL import ImageEnhance
from sklearn.preprocessing import label_binarize


#%%
# Rename your 'honey-bee-annotated-images' to 'data' and put your own current working path
cwd = r'C:\_leshan\school\M2SID\DL\BeeImage\data\\'
#cwd = r'C:\Users\leobo\OneDrive\Documents\Cours\DeepLearning\Projet\Dataset\\'
os.chdir(cwd)
imgenhancement = False


#%%  Input csv
csv_path = cwd + 'bee_data.csv' 
Y = pd.read_csv(csv_path)
Y = Y[['file', 'zip code', 'subspecies', 'health']] # removing useless lines
Y = Y[Y['subspecies']!='-1'] # Deleting no-label Y
Y.sort_values('file', axis = 0, ascending = True, inplace = True, na_position ='last')
Y = Y.reset_index(drop=True)


#%%  Input images for seeing shapes
img_path = cwd + 'bee_imgs\\bee_imgs\\' 
img_files = Y['file']# List of String
#imgs = imgs[:200] # saving time during the test
images = np.array([imageio.imread(img_path+img) for img in img_files]) # array of all images
max_shape = np.amax(np.array([i.shape for i in images]), axis=0) # like: array([152, 161, 3])


#%%  Selecting max_shape with a threshold
# This part for checking the X shape and delete the useless images in X and Y
# By using the global max_shape, the X tensor has 5172*520*392*3 = 3 GB
pixels_threshold = 100

shapes = np.array([i.shape for i in images])
plt.plot(np.bincount(shapes[:,0]), label='length pixels')
plt.plot(np.bincount(shapes[:,1]), label='width pixels')
plt.title('Images size distribution'); plt.legend(); plt.show()

max_shape_per_img = np.max(shapes, axis=1)
Y['longest side pixels'] = max_shape_per_img 
Y = Y[Y['longest side pixels']<=pixels_threshold]
Y.sort_values('file', axis = 0, ascending = True, inplace = True, na_position ='last')
Y = Y.reset_index(drop=True)


#%%  Re-input images with Enhancement
img_files = Y['file'] # List of String
cspace = 'HSV'


def enhancement(im_path, brightness=1, color=2, cspace='RGB'):
    image = imageio.imread(im_path, pilmode=cspace)
    image = Image.fromarray(image)
    enh_bri = ImageEnhance.Brightness(image)
    image_brightened = enh_bri.enhance(brightness)
    enh_col = ImageEnhance.Color(image_brightened)
    image_colored = enh_col.enhance(color)
    if cspace == 'HSV':
        image_colored.convert('HSV')
    im1 = np.array(image_colored) 
    return im1

if imgenhancement:
    images = np.array([enhancement(img_path+img, 1, 2, cspace) for img in img_files])        
else:
    images = np.array([imageio.imread(img_path+img, pilmode=cspace) for img in img_files]) # array of all images
        
max_shape = np.amax(np.array([i.shape for i in images]), axis=0) 
    
    
#%%  Zero-padding
def zero_padding(image, max_shape, cspace):
    image = image[:,:,0:3] # Take only the first 3 color dimensions
    delta_w = max_shape[1] - image.shape[1] 
    delta_h = max_shape[0] - image.shape[0]
    top, bottom, left, right = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
    if cspace == 'RGB':
        image_re = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255]*3)
    elif cspace == 'HSV':
        image_re = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 255])
    return image_re   

X = np.array([zero_padding(image, max_shape, cspace) for image in images]) # treating all images


#%%  Saving to tensors
if imgenhancement:
    np.save('Xenhbee%s.npy'%cspace, X)
    np.save('Yenhbee%s.npy'%cspace, Y)
else:
    np.save('Xbee%s.npy'%cspace, X)
    np.save('Ybee%s.npy'%cspace, Y)


#%%  Checking X Y consistency
for i in range(1, len(Y), 1000):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(X[i])
    axarr[0].set_title("Padded Enhanced Image in X in %s"%cspace)  
#    axarr[0].axis('off')
    axarr[1].imshow(imageio.imread(img_path+Y['file'][i]))
    axarr[1].set_title("Related Images in Y['file']")  
#    axarr[1].axis('off')
    plt.show()
print('X shape : ', X.shape)
print('Y shape : ', Y.shape)
print('Y labels:  %s'%' | '.join(list(Y)))
    

#%%  Trial
#i = 150
#a = images[i]
#A = zero_padding(a, max_shape)
#plt.imshow(a);plt.show()
#plt.imshow(A);plt.show()


#%%
