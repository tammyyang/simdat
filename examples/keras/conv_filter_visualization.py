'''Visualization of the filters of VGG16, via gradient ascent in input space.

This script can run on CPU in a few minutes (with the TensorFlow backend).

Results example: http://i.imgur.com/4nj4KjN.jpg

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file.
'''
from __future__ import print_function
from scipy.misc import imsave
import numpy as np
import time
import os
import h5py
from simdat.core import dp_models
from keras.models import Sequential
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras import backend as K

dp = dp_models.DPModel()
# dimensions of the generated pictures for each filter.
img_width = 224
img_height = 224
# img_width = 128
# img_height = 128

# path to the model weights file.
weights_path = '/home/tammy/www/vgg-16/vgg16_weights.h5'

# the name of the layer we want to visualize (see model definition below)
output_index = 65

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# build the VGG16 network
model = dp.VGG_16(weights_path=weights_path)
model.summary()
input_img = model.layers[0].input
# this is a placeholder tensor that will contain our generated images
print('Model loaded.')


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


kept_filters = []
start_time = time.time()

layer_output = model.layers[-5].output
loss = K.mean(layer_output[:, output_index])

# we compute the gradient of the input picture wrt this loss
grads = K.gradients(loss, input_img)[0]

# normalization trick: we normalize the gradient
grads = normalize(grads)

# this function returns the loss and grads given the input picture
# Keras.backend.function(inputs, outputs)
#     inputs: list of placeholder/variable tensors.
#     outputs: list of output tensors.
iterate = K.function([input_img], [loss, grads])

# step size for gradient ascent
step = 1.

# we start from a gray image with some random noise
# input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 128.
input_img_data = np.random.random((1, 3, img_width, img_height)) * 20 + 224.

# we run gradient ascent for 20 steps
for i in range(20):
    loss_value, grads_value = iterate([input_img_data])
    input_img_data += grads_value * step

    print('Current loss value:', loss_value)
    if loss_value <= 0.:
        # some filters get stuck to 0, we can skip them
        break

# decode the resulting input image
if loss_value > 0:
    img = deprocess_image(input_img_data[0])
end_time = time.time()
print('Processed in %ds' % (end_time - start_time))

# save the result to disk
imsave('stitched_filters.png', img)
