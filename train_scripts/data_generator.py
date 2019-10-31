import tensorflow as tf
from tensorflow.keras import layers
import os

 

# In[2]:


import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
 #import keras

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images

from skimage.measure import compare_psnr, compare_ssim


from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

#import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout, UpSampling2D
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img



from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers.merge import Add
from keras.layers.core import Dropout
#import tensorflow as tf

from keras.models import Model
from keras.engine import InputSpec
from keras.engine.topology import Layer
from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers.merge import Add
from keras.utils import conv_utils
from keras.layers.core import Dropout

from keras.layers import Input, Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model


from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model

 


 
 

import tensorflow as tf
import keras.backend as K
from keras.applications.vgg16 import VGG16
from keras.models import Model
from  keras.layers import Concatenate
 

 

def get_data(path='train/', train=True):
    ids = next(os.walk(path + "raw"))[2]
    
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load raw
        img = load_img(path + 'raw/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (448,896, 1), mode='constant', preserve_range=True)

        # Load average
        if train:
            average = img_to_array(load_img(path + 'average/' + id_, grayscale=True))
            average = resize(average, (448, 896, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = average / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X
    
  

 
 
def extract_patches(image):
    import tensorflow as tf
 
    ksize_rows = 40
    ksize_cols = 60
    strides_rows = 10
    strides_cols = 10

    sess = tf.InteractiveSession()

 
    # The size of sliding window
    ksizes = [1, ksize_rows, ksize_cols, 1] 

    # How far the centers of 2 consecutive patches are in the image
    strides = [1, strides_rows, strides_cols, 1]

 
    rates = [1, 1, 1, 1] # sample pixel consecutively

    # padding algorithm to used
    padding='VALID' # or 'SAME'

    image = tf.expand_dims(image, 0)
    #image = image.reshape((1,448,896,1))
    #print(image.shape)
    image_patches = tf.extract_image_patches(image, ksizes, strides, rates, padding)

    # print image shape of image patches
    #print(sess.run(tf.shape(image_patches)))

    # image_patches is 4 dimension array, you can use tf.squeeze to squeeze it, e.g.
    # image_patches = tf.squeeze(image_patches)

    
    # retrieve the 1st patches
    patch1 = image_patches[0,0,0,]
    patch2 = image_patches[0,0,10,]

    # reshape
    patches = tf.reshape(image_patches, [image_patches.shape[1],image_patches.shape[2],ksize_rows, ksize_cols,1])
    patch1 = patches[0,:,:,0]
    #patch2 = tf.reshape(patch2,[40,60])
    #print(patch1.shape)
    #scipy.misc.imsave('GAN/patches/i1.png',sess.run(patch2 ))
    # visualize image
    patches = sess.run(patches)
    '''
    for i in range(50):
        scipy.misc.imsave('GAN/patches/'+str(i)+'.png',patches[0,i,:,:,0])
        #scipy.misc.imsave('GAN/patches/'+str(i)+'_test.png',patches[i,:,:,0])
 
        if i==5:
            break
    '''
    sess.close()
    patches = patches.reshape(patches.shape[0]*patches.shape[1],patches.shape[2],patches.shape[3],patches.shape[4])
    print(patches.shape)
    
    return patches

def get_data_patches(path='train/'):
    import os
    ids = next(os.walk(path + "raw"))[2]
    
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
 
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load raw
        img = load_img(path + 'raw/' + id_, color_mode = "grayscale")
        x_img = img_to_array(img)
        x_img = resize(x_img, (448,896, 1), mode='constant', preserve_range=True)
 
        
        if n==0:
            X_patches = extract_patches (x_img )
        else:
            np.append(X_patches,extract_patches (x_img ))
        
        # Load average
 
        average = img_to_array(load_img(path + 'average/' + id_, color_mode = "grayscale"))
        average = resize(average, (448, 896, 1), mode='constant', preserve_range=True)
        #average = average/255
        if n==0:
            y_patches = extract_patches (average )
        else:
            np.append(y_patches, extract_patches (average ))

        # Save images
        X[n] = x_img  / 255
 
        y[n] = average / 255
        #print(X_patches.shape)
        #if n == 2:
        #scipy.misc.imsave('GAN/i1.png',X_patches[160,:,:,0])
        #scipy.misc.imsave('GAN/i2.png',y_patches[160,:,:,0])
        #scipy.misc.imsave('GAN/i1.png',X_patches[0,:,:,0])
        #scipy.misc.imsave('GAN/i2.png',y_patches[0,:,:,0])

        #    break
    print('Done!')
 
    return X_patches , y_patches
 

