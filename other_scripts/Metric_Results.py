
# coding: utf-8

# In[1]:


from PIL import Image
import scipy.stats  
import numpy as np
 


# In[2]:


for i in range(10):
    im = Image.open('train/average/'+str(i+1)+'.tif')
    snr =  np.mean(im)/np.std(im)
    print(snr)


# In[ ]:


import cv2 
#for i in range(10):
im = cv2.imread('train/average/1.tif',0)
 
#r = cv2.selectROI(im)
#cv2.imshow('image', r)
#r = [10,50,50,50]    
    # Crop image
#imCrop = im[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # Display cropped image
print(im)
cv2.startWindowThread()
cv2.namedWindow("preview")
cv2.imshow("preview", im)

cv2.waitKey()


# In[5]:


import numpy 
import math
import cv2
original = cv2.imread("GAN/SBSDI_results/average1.png")
contrast = cv2.imread("GAN/SBSDI_results/results1.png")

def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = numpy.amax(img1)
    print(PIXEL_MAX)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

d=psnr(original,contrast)
print(d)

