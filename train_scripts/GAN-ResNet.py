
# coding: utf-8

# In[10]:


import tensorflow as tf
from tensorflow.keras import layers
import os

print(tf.VERSION)
print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.Session(config=config)

set_session(sess)  # set this TensorFlow session as the default session for Keras


# In[11]:


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


from keras import backend as K
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
 
from keras.models import Sequential, Model

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

from keras.layers import Input
from keras.models import Model
import os
import datetime
import click
import numpy as np
import tqdm
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
os.chdir("../")



from Models.models import *



input_shape = np.array([100,200,1])
ngf = 64
ndf = 64 
n_blocks_gen = 16
path_train = 'dataset/train/'
epochs = 25

#Set True to train using combined perceptual and MSSSIM loss
MSSSIM = True


# In[12]:

 
def generator_containing_discriminator_multiple_outputs(generator, discriminator):
    inputs = Input(shape=input_shape)
    generated_images = generator(inputs)
    outputs = discriminator(generated_images)
    model = Model(inputs=inputs, outputs=[generated_images, outputs])
    return model


# In[18]:



def perceptual_loss(y_true, y_pred):
    vgg = VGG16(include_top=False,   input_shape=(input_shape[0],input_shape[1],3) )
    y_true_c =   Concatenate()([y_true,y_true,y_true])    
    y_pred_c =   Concatenate()([y_pred,y_pred,y_pred])    

    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true_c) - loss_model(y_pred_c)))

def MSSIM_loss(y_true, y_pred):
    # source: https://gist.github.com/Dref360/a48feaecfdb9e0609c6a02590fd1f91b
    vgg = VGG16(include_top=False,   input_shape=(input_shape[0],input_shape[1],3) )
    y_true_c =   Concatenate()([y_true,y_true,y_true])    
    y_pred_c =   Concatenate()([y_pred,y_pred,y_pred])    

    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False
    p_loss = K.mean(K.square(loss_model(y_true_c) - loss_model(y_pred_c)))
    s_loss = -1*tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    if MSSSIM:
        loss =  p_loss+s_loss
    else:
        loss = p_loss
    return loss
    

def wasserstein_loss(y_true, y_pred):
    #return K.mean(K.square(y_true)-K.square(y_pred))
    return K.mean(y_true*y_pred)


# In[19]:

def extract_patches(image):
    import tensorflow as tf
    print(input_shape) 
    ksize_rows = input_shape[0]
    ksize_cols = input_shape[1]
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
    #patch1 = image_patches[0,0,0,]
    #patch2 = image_patches[0,0,10,]

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



def get_patch_data(path,train=True):
    print(input_shape)
    ids = next(os.walk(path + "raw"))[2]
    im_height,im_width,nc = (448,896, 1)
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
 
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load raw
        img = load_img(path + 'raw/' + id_, color_mode = "grayscale")
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
 
        
        if n==0:
            X_patches = extract_patches (x_img )
        else:
            np.append(X_patches,extract_patches (x_img ))
        
        # Load average
 
        average = img_to_array(load_img(path + 'average/' + id_, color_mode = "grayscale"))
        average = resize(average, (im_height, im_width, 1), mode='constant', preserve_range=True)
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
 
def get_data(path, train=True):
    print(input_shape)
    ids = next(os.walk(path + "raw"))[2]
    im_height,im_width,nc = (448,896, 1)
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load raw
        img = load_img(path + 'raw/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)

        # Load average
        if train:
            average = img_to_array(load_img(path + 'average/' + id_, grayscale=True))
            average = resize(average, (im_height, im_width, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = average / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X


# In[21]:



def save_all_weights(d, g, epoch_number, current_loss):
    now = datetime.datetime.now()
    save_dir = os.path.join('GAN/', '{}{}'.format(now.month, now.day))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    g.save_weights(os.path.join(save_dir, 'generator_{}_{}.h5'.format(epoch_number, current_loss)), True)
    d.save_weights(os.path.join(save_dir, 'discriminator_{}.h5'.format(epoch_number)), True)
    #s.save_weights(os.path.join(save_dir, 'siamese_{}.h5'.format(epoch_number)), True)


def train_multiple_outputs(input_shape,n_images, batch_size, log_dir, epoch_num, critic_updates=5):
    print(input_shape)
    g = generator_model(ngf = ngf,input_nc = input_shape[2],output_nc = input_shape[2],input_shape = input_shape,n_blocks_gen = n_blocks_gen )
    #g =build_res_unet()
    #g.load_weights('GAN/16_RES/410/generator_69_36.h5')

    d = discriminator_model(ndf = ndf,output_nc = input_shape[2],input_shape = input_shape)
    d_on_g = generator_containing_discriminator_multiple_outputs(g, d)

    d_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    d_on_g_opt = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    d.trainable = True
    d.compile(optimizer=d_opt, loss=wasserstein_loss)
    d.trainable = False
    loss = [perceptual_loss, wasserstein_loss]
    loss_weights = [100, 5]
    d_on_g.compile(optimizer=d_on_g_opt, loss=loss, loss_weights=loss_weights)
    d.trainable = True

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), -np.ones((batch_size, 1))

    log_path = './logs'
    tensorboard_callback = TensorBoard(log_path)

    for epoch in tqdm.tqdm(range(epoch_num)):
        permutated_indexes = np.random.permutation(x_train.shape[0])

        d_losses = []
        d_on_g_losses = []
        for index in range(int(x_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_blur_batch = x_train[batch_indexes]
            image_full_batch = y_train[batch_indexes]

            generated_images = g.predict(x=image_blur_batch, batch_size=batch_size)

            for _ in range(critic_updates):
                d_loss_real = d.train_on_batch(image_full_batch, output_true_batch)
                d_loss_fake = d.train_on_batch(generated_images, output_false_batch)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                d_losses.append(d_loss)

            d.trainable = False

            d_on_g_loss = d_on_g.train_on_batch(image_blur_batch, [image_full_batch, output_true_batch])
            d_on_g_losses.append(d_on_g_loss)

            d.trainable = True

        #write_log(tensorboard_callback, ['g_loss', 'd_on_g_loss'], [np.mean(d_losses), np.mean(d_on_g_losses)], epoch_num)
        print( np.mean(d_on_g_losses))
        with open('log.txt', 'a+') as f:
            f.write('{} - {} - {}\n'.format(epoch, np.mean(d_losses), np.mean(d_on_g_losses)))

        save_all_weights(d, g, epoch, int(np.mean(d_on_g_losses)))
  

    
x_train, y_train = get_data(path_train, train=True)
n_images = x_train.shape[0]  
train_multiple_outputs(input_shape,n_images, batch_size=1, log_dir="lg", epoch_num=epochs, critic_updates=5)


