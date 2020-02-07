
# coding: utf-8

# In[ ]:

from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import os

print(tf.VERSION)
print(tf.__version__)
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.Session(config=config)

set_session(sess)  # set this TensorFlow session as the default session for Keras


# In[18]:

import imageio
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

#import keras

from tqdm import tqdm_notebook, tnrange
from itertools import chain
import datetime
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
cwd = os.getcwd()

from Models.models import *
input_shape = np.array([448,896,1])

ngf = 64
ndf = 64 
n_blocks_gen = 16

#Give path to the pretrained model
now = datetime.datetime.now()
saved_dir = os.path.join('GAN/', '{}{}'.format(now.month, now.day))
model_path = cwd + "/" + saved_dir
# In[19]:

 

# In[24]:


im_width = 896
im_height = 448
border = 5
path_train = cwd + '/dataset/train/'
path_test = cwd + '/dataset/test/'

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
# from PIL import Image
import click
import scipy.misc
import statistics
# from scipy import stats

def signaltonoise(a, axis, ddof): 
    a = np.asanyarray(a) 
    m = a.mean(axis) 
    sd = a.std(axis = axis, ddof = ddof) 
    return np.where(sd == 0, 0, m / sd) 


def test(batch_size):
    max_val = 0
    ids = next(os.walk(model_path))[2]

    g = generator_model(ngf = ngf,input_nc = input_shape[2],output_nc = input_shape[2],input_shape = input_shape,n_blocks_gen = n_blocks_gen)
    #g =build_res_unet()
    for temp in ids:
        if "generator_" in temp:
            p=[]
            q=[]
            q_1=[]
            r=[]
            temp = model_path + "/" + temp
            g.load_weights(temp)
            generated_images = g.predict(x=x_test, batch_size=batch_size)
            #generated = np.array([deprocess_image(img) for img in generated_images])
            #x_test = deprocess_image(x_test)
            #y_test = deprocess_image(y_test)
           
            for i in range(generated_images.shape[0]):
                y = y_test[i, :, :, :]
                x = x_test[i, :, :, :]
                img = generated_images[i, :, :, :]
                output = np.concatenate((y, x, img), axis=1)

                p.append(compare_psnr(img[:,:,0],y[:,:,0]))
                q.append(compare_ssim(img[:,:,0],y[:,:,0]))
                q_1.append(compare_ssim(img[:,:,0],y[:,:,0], multichannel=True))
                # q_1.append(tf.image.ssim(img,y, 1.0))
                # r.append(stats.signaltonoise(img[:,:,0]))
                r.append(signaltonoise(img[:,:,0], axis = 0, ddof = 0))
                #print(p)
            psnr = np.array(p).mean()
            if psnr > max_val:
                max_val = psnr
                with open(model_path + "/logs.txt", "a") as f:    
                    for value in p:
                        f.write(str(value) + " ")
                    f.write("\n")
                for i in range(generated_images.shape[0]):
                    y = y_test[i, :, :, :]
                    x = x_test[i, :, :, :]
                    img = generated_images[i, :, :, :]
                    output = np.concatenate((y, x, img), axis=1)
                    # imageio.imwrite('filename.jpg', array)
                    imageio.imwrite(cwd + '/GAN/Results/results{}.png'.format(i+1),img[:,:,0])
                    imageio.imwrite(cwd + '/GAN/Results/raw{}.png'.format(i+1),x[:,:,0])
                    imageio.imwrite(cwd + '/GAN/Results/average{}.png'.format(i+1),y[:,:,0])

            with open(cwd + '/GAN/Results' + "/logs.txt", "a") as f:    
                f.write("generator is " + str(temp) + "\n")
                f.write("psnr = " + str(np.array(p).mean()) + "\n")
                f.write("Standard Deviation of psnr is % s " % (statistics.stdev(p)) + "\n") 
                f.write("ssim = " + str(np.array(q).mean()) + "\n")
                # f.write("ssim_! = " + str(np.array(q_1).mean()) + "\n")
                f.write("current PSNR value is = " + str(max_val) + "\n\n")

            print("generator is " + str(temp))
            print("psnr = " + str(np.array(p).mean()))
            print("Standard Deviation of psnr is % s " % (statistics.stdev(p)))
            print("ssim = " + str(np.array(q).mean()))
            # print("ssim_! = " + str(np.array(q_1).mean()))
            print("current max is = " + str(max_val))

        # im = Image.fromarray(output.astype(np.uint8))
        # im.save('results{}.png'.format(i))
    


test(1)

# with open('GAN/results.txt','a') as f:
#     f.write('16_RES_RESULTS\n')
#     for psnr in p:
#         f.write(str(psnr)+'\n')


# In[10]:


# print(p)
# print(q)

