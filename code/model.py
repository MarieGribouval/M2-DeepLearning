
'''
# Created on Thu Nov 14 10:20:58 2019
# @title : Deep Bees - Model
# @author: Marie Gribouval, Léo Boule and Leshanshui YANG
'''

import keras

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten
from keras.models import Model

from keras.optimizers import Adam

#Model with 2 convs
def model(input_size,output_size):
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


