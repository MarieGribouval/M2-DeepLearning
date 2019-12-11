import keras
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

import matplotlib.pyplot as plt
import seaborn as sns


def net(input_size,output_size):
    print(input_size)
    inputs = Input(shape=input_size)
    model = Conv2D(9,9,activation="relu",padding='same') (inputs)
    model = MaxPooling2D(pool_size=(2,2)) (model)
    model = Conv2D(7,7,activation="relu",padding="same") (model)
    model = MaxPooling2D(pool_size=(2,2)) (model)
    model = Flatten()(model)
    output = Dense(output_size[1],activation="softmax") (model)

    model = Model(input=inputs,outputs = output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


#%  Preparing inputs
import os 
os.chdir(r'C:\_leshan\school\M2SID\DL\BeeImage\data\\')
import numpy as np
img = np.load('XenhbeeRGB.npy')[105,:,:]
plt.imshow(img); plt.title('photo input'); plt.show()
import tensorflow as tf
t = tf.convert_to_tensor(img, tf.float32, name='t')
t = tf.reshape(t, (1, 100, 100, 3), name=None)


#%  Preparing outputs of each layer
network = net((100,100,3), (1,1))
main_layers = network.layers[1:5]
outputs = [layer.output for layer in main_layers]
comp_graph = [K.function([network.input]+ [K.learning_phase()], [output]) for output in outputs]
layer_outputs_list = [op([t, 1.]) for op in comp_graph]
# code reference of this block: https://cloud.tencent.com/developer/ask/83195

# Change model output into list of array and print shape   
layer_outputs = []
layer_names = [layer.name for layer in main_layers]
for name, layer_output in zip(layer_names, layer_outputs_list):
    print(name, '\nshape:', layer_output[0][0].shape, end='\n-------------------\n')
    layer_outputs.append(layer_output[0][0])


#%  Plotting
def plot_layer_outputs(layer_number, layer_outputs, layer_name=''):    
    x_max = layer_outputs[layer_number].shape[0]
    y_max = layer_outputs[layer_number].shape[1]
    n = layer_outputs[layer_number].shape[2]
    L = []
    for i in range(n):
        L.append(np.zeros((x_max, y_max)))
    for i in range(n):
        for x in range(x_max):
            for y in range(y_max):
                L[i][x][y] = layer_outputs[layer_number][x][y][i]
    fig = plt.figure(figsize=[12, 6])
    fig.suptitle('_'.join(layer_name.split('_')[:-1]), fontsize=10)
    nb_w = 1
    nb_h = n//nb_w + 1
    for i, img in enumerate(L):
        plt.subplot(nb_h, nb_w, i+1)
        plt.imshow(img, interpolation='nearest', cmap='gray')


for position, name in enumerate(layer_names):
    plot_layer_outputs(position, layer_outputs, name)  



#%  Reading h5
import h5py
filename = 'models\model_XenhbeeHSV_subs.h5'
with h5py.File(filename, 'r') as f:
    print("Keys: %s" % f.keys())

    
#%  Loading model
from keras.models import load_model
network = load_model('models\model_XenhbeeHSV_subs.h5')

# For the 1st conv layer:
#for i in range(int(network.layers[1].weights[0].shape[3]):

ws = []    
for l in network.layers:
    try:
        ws.append(l.weights[0])
    except:
        pass
#vars(network.layers[1].weights[0])

[:, :, :, i]

