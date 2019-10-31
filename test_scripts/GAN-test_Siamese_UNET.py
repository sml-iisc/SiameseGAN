
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers
import os

print(tf.VERSION)
print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.Session(config=config)

set_session(sess)  # set this TensorFlow session as the default session for Keras


# In[18]:


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
os.chdir("../")


from Models.models import *
input_shape = np.array([448,896,1])

ngf = 64
ndf = 64 
n_blocks_gen = 16

#Give path to the pretrained model
model_path = 'GAN/generator.h5'

 
# In[19]:

 

# In[24]:


im_width = 896
im_height = 448
border = 5
path_train = 'dataset/train/'
path_test = 'dataset/test/'

def get_data(path, train=True):
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
    
x_test, y_test = get_data(path_test, train=True)


# In[ ]:


import numpy as np
from PIL import Image
import click
import scipy.misc
#from scipy.stats import signaltonoise
p=[]
def test(batch_size):
    #g = generator_model(ngf = ngf,input_nc = input_shape[2],output_nc = input_shape[2],input_shape = input_shape,n_blocks_gen = n_blocks_gen)
    g =build_res_unet(input_shape = input_shape)

    g.load_weights(model_path)
    generated_images = g.predict(x=x_test, batch_size=batch_size)
    #generated = np.array([deprocess_image(img) for img in generated_images])
    #x_test = deprocess_image(x_test)
    #y_test = deprocess_image(y_test)
   
    for i in range(generated_images.shape[0]):
        y = y_test[i, :, :, :]
        x = x_test[i, :, :, :]
        img = generated_images[i, :, :, :]
        output = np.concatenate((y, x, img), axis=1)
        scipy.misc.imsave('GAN/Results/results{}.png'.format(i+1),img[:,:,0])
        scipy.misc.imsave('GAN/Results/raw{}.png'.format(i+1),x[:,:,0])
        scipy.misc.imsave('GAN/Results/average{}.png'.format(i+1),y[:,:,0])

        p.append(compare_psnr(img[:,:,0],y[:,:,0]))
        #snr = signaltonoise(img[:,:,0])
        #print(p)
    print(np.array(p).mean())
 

        #im = Image.fromarray(output.astype(np.uint8))
        #im.save('results{}.png'.format(i))

test(1)

with open('GAN/results.txt','a') as f:
    f.write('16_RES_RESULTS\n')
    for psnr in p:
        f.write(str(psnr)+'\n')


# In[10]:


print(p)

