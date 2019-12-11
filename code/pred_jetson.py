#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Inspired by Chengwei's tutorial
#https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/

output_names = ['dense_2/Softmax']
input_names = ['input_2']

import tensorflow as tf


def get_frozen_graph(graph_file):
    """Read Frozen Graph file from disk."""
    with tf.gfile.FastGFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


trt_graph = get_frozen_graph('/home/insajetson/BeeProject/trt_graph_model_XbeeHSV_heal.pb')

# Create session and load graph
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf_sess = tf.Session(config=tf_config)
tf.import_graph_def(trt_graph, name='')


# Get graph input size
for node in trt_graph.node:
    if 'input_' in node.name:
        size = node.attr['shape'].shape
        image_size = [size.dim[i].size for i in range(1, 4)]
        break
print("image_size: {}".format(image_size))


# input and output tensor names.
input_tensor_name = input_names[0] + ":0"
output_tensor_name = output_names[0] + ":0"

print("input_tensor_name: {}\noutput_tensor_name: {}".format(
    input_tensor_name, output_tensor_name))

output_tensor = tf_sess.graph.get_tensor_by_name(output_tensor_name)


# In[3]:


from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np


X = np.load("/home/insajetson/BeeProject/XbeeHSV_test_health.npy")
X = X.astype('float32')
X/=255
print(X.shape)
X = X[0:20]
#X = np.expand_dims(X,axis=0)
print(X.shape)
X = preprocess_input(X)

feed_dict = {
    input_tensor_name: X
}
preds = tf_sess.run(output_tensor, feed_dict)


# In[4]:


import time
times = []
for i in range(20):
    start_time = time.time()
    one_prediction = tf_sess.run(output_tensor, feed_dict)
    delta = (time.time() - start_time)
    times.append(delta)
mean_delta = np.array(times).mean()
fps = 1 / mean_delta
print('average(sec):{:.2f},fps:{:.2f}'.format(mean_delta, fps))


# In[ ]:




