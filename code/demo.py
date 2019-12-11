# -*- coding: utf-8 -*-
'''
# Created on Wed Dec 11 22:03:39 2019
# @title : Deep Bees - Demonstration
# @author: Marie Gribouval, Léo Boule and Leshanshui YANG
'''


model_path = r'C:\_leshan\school\M2SID\DL\BeeImage\code\\'
import os
os.chdir(model_path)
import numpy as np
import pandas as pd
import cv2
import imageio
import matplotlib.pyplot as plt
from keras.models import load_model


def zero_padding(image, max_shape, cspace='RGB'):
    image = image[:,:,0:3] # Take only the first 3 color dimensions
    delta_w = max_shape[1] - image.shape[1] 
    delta_h = max_shape[0] - image.shape[0]
    top, bottom, left, right = delta_h//2, delta_h-(delta_h//2), delta_w//2, delta_w-(delta_w//2)
    if cspace == 'RGB':
        image_re = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255]*3)
    elif cspace == 'HSV':
        image_re = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 255])
    return image_re   


#%%
heal_model = load_model(model_path + '\\model_XenhbeeRGB_heal.h5')
#subs_model = load_model(model_path + '\\model_XenhbeeRGB_subs.h5')
path = r'C:\_leshan\school\M2SID\DL\BeeImage\tests\\'
csv_path = path + 'bee_data.csv' 
path += r'images\\'
Y = pd.read_csv(csv_path)
#path="C:/Users/leobo/OneDrive/Documents/Cours/DeepLearning/Projet/Dataset/Demo_dataset/"
Ximg = [f for f in os.listdir(path)]
Ximg.sort()


#%%  Showing raw
images = []
for i in range(len(Ximg)):
    lo = imageio.imread(path+Ximg[i])
    images.append(lo)
    plt.subplot(2, 5, i+1); plt.imshow(lo); plt.title(Ximg[i])
plt.show()

print('-'*16, '\nProcessing padding......')
X = np.array([zero_padding(image, max_shape=(100,100), cspace='RGB') for image in images]) 
print('X shape after padding :', X.shape)


# Predicting and Testing
ind = [ Y[Y['file']== i].index[0] for i in Ximg ]
Y_real = Y.loc[ind]
groundtruth = list(Y_real['health'])

Y_predheal = np.argmax(heal_model.predict(X,verbose=1), axis=1)
heal_dict = {0 : 'Varroa, Small Hive Beetles', 1 : 'ant problems', 2 : 'few varrao, hive beetles', 3 : 'healthy'}
preds = [heal_dict[i] for i in Y_predheal]

testn = 0
print('-'*16, '\nPredicting......')
for i, j in zip(groundtruth, preds):
    print('Test %2d :    ✔'%testn if i==j else 'Test %2d :    ×'%testn)
    print('Predicted    : ', j)
    print('Ground Truth : ', i)
    testn += 1


