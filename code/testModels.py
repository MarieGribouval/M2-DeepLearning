# -*- coding: utf-8 -*-
'''
# Created on Thu Nov 14 10:34:58 2019
# @title : Deep Bees - Data Extraction and Cleansing
# @author: Marie Gribouval, Léo Boule and Leshanshui YANG
'''


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedShuffleSplit
from keras.callbacks import ModelCheckpoint
from keras.callbacks.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns


#%%  Loading data
#cwd = 'C:/Users/leobo/OneDrive/Documents/Cours/DeepLearning/Projet/Dataset//'
cwd = 'C:\_leshan\school\M2SID\DL\BeeImage\data\\'
os.chdir(cwd)
import model

Xname = 'XbeeRGB' # Or could be XenhbeeRGB or XenhbeeHSV or XbeeHSV
X = np.load(cwd + Xname + '.npy')
X = X.astype('float32')
X /= 255

Yraw = pd.DataFrame(np.load('Ybee.npy', allow_pickle=True), 
                    columns=['file', 'zip code', 'subspecies', 'health', 'Images size distribution']).astype('str')
models, acc_list, wrong_ind, wrong_list = [], [], [], []
ycols = ['subspecies', 'health']


# For each property we create a Y
for ycol in ycols:
    Yclasses = list(Yraw.groupby([ycol]).count().index)
    print('Y property of        :\n   ', ycol)
    print('Y classes (by order) :\n   ', ' || '.join(Yclasses))
    Y = Yraw[ycol]
    Y = label_binarize(Y, classes=Yclasses)
    
    #%  Splitting data sets
    splt = StratifiedShuffleSplit(1, test_size=0.30,random_state=0)
#    print(splt.get_n_splits(X,Y))
    for train_index, test_index in splt.split(X, Y):
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
 
    # By using validation_split = 0.1 in model.fit(), no use to split manually
#    Ntrain = int((Xtrain.shape[0]/10)*9)
#    Nvalid = int(Xtrain.shape[0] - Ntrain)
#    Xvalid = Xtrain[Ntrain:]
#    Xtrain = Xtrain[0:Ntrain]
#    Yvalid = Ytrain[Ntrain:]
#    Ytrain = Ytrain[0:Ntrain]
#    Y = Y[['zip code', 'subspecies', 'health']]
    
    batch_size = 8
    epoch = 100
    #print(Xtrain.shape)
    #print(Ytrain.shape)
    
    bee_model = model.model((Xtrain.shape[1],Xtrain.shape[2],Xtrain.shape[3]),(Ytrain.shape[0],Ytrain.shape[1]))
    #print(bee_model.summary())
    checkpoint = ModelCheckpoint('models\\model_%s_%s.h5'%(Xname, ycol[:4]), verbose=1, monitor='val_loss', 
                                 save_best_only=True, mode='auto')  
    earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto')
    mod_fitted = bee_model.fit(Xtrain, Ytrain, batch_size=batch_size, epochs=epoch, 
                               validation_split=0.1, callbacks=[checkpoint, earlyStopping], verbose=False)

    loss_train = mod_fitted.history['loss']
    loss_valid = mod_fitted.history['val_loss']
    metric_train = mod_fitted.history['accuracy']
    metric_valid = mod_fitted.history['val_accuracy']
    
    plt.plot(loss_train,"b:o", label = "loss_train")
    plt.plot(loss_valid,"r:o", label = "loss_valid")
    plt.title("Loss over training epochs : %s"%ycol)
    plt.legend()
    plt.show()
    print("Train finished")
    
    #%  Testing
    Y_predict = bee_model.predict(Xtest,batch_size=batch_size,verbose=1)
#    fig = plt.figure(figsize=[3,9]); sns.heatmap(Y_predict); plt.show()
    Y_predict = np.argmax(Y_predict, axis=1)
    Y_test = np.argmax(Ytest, axis=1)
    
    acc = np.sum(Y_predict==Y_test)/Y_test.shape[0]
    print('Accuracy : %.4f'%acc)
    # We mistakenly predicted aa as bb, so aa->bb
    error_ind = np.where(Y_predict!=Y_test)[0]
    error_list = np.array([str(Y_test[i])+'->'+str(Y_predict[i]) for i in np.where(Y_predict!=Y_test)[0]]).T.reshape(-1)
    error_list = pd.DataFrame(error_list, columns=['real -> pred'])
#    print('Wrong predictions', error_list.groupby('real -> pred').size())
    
    models.append(bee_model)
    acc_list.append(acc)
    wrong_ind.append(error_ind)
    wrong_list.append(error_list)


for ycol, accuracy in zip(ycols, acc_list):
    print('Accuary on %s:\n    %.2f%%'%(ycol,accuracy*100))
  
#% The Correct Prediction over all properties:
nb_has_wrong = len(set(np.concatenate([i for i in wrong_ind])))
print('2-properties all correct accuracy: %.2f%%'%(100-nb_has_wrong/len(Y_predict)*100))

