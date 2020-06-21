from PIL import Image
import numpy as np
import os
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.measure import compare_psnr, compare_ssim

ddepth = cv.CV_16S
kernel_size = 3

os.chdir("../")
cwd = os.getcwd()
path = cwd + '/Results/dataset2/'
# path = cwd + '/dataset/test/real/Results/'
# path = "/home/nilesh2019/workspace/image_denoise/GAN/Results/"

image1   = path + 'results1.png'
image2   = path + 'results2.png'
image3   = path + 'results3.png'
image4   = path + 'results4.png'
image5   = path + 'results5.png'
image6   = path + 'results6.png'
image7   = path + 'results7.png'
image8   = path + 'results8.png'
image9   = path + 'results9.png'
image10  = path + 'results10.png'
image11  = path + 'results11.png'
image12  = path + 'results12.png'
image13  = path + 'results13.png'
image14  = path + 'results14.png'
image15  = path + 'results15.png'
image16  = path + 'results16.png'
image17  = path + 'results17.png'
image18  = path + 'results18.png'
image19  = path + 'results19.png'
image20  = path + 'results20.png'

# resnet
# image1  = path + 'resnet/results1.png'
# image2  = path + 'resnet/results2.png'
# image3  = path + 'resnet/results3.png'
# image4  = path + 'resnet/results4.png'
# image5  = path + 'resnet/results5.png'
# image6  = path + 'resnet/results6.png'
# image7  = path + 'resnet/results7.png'
# image8  = path + 'resnet/results8.png'
# image9  = path + 'resnet/results9.png'
# image10 = path + 'resnet/results10.png'
# image11 = path + 'resnet/results11.png'
# image12 = path + 'resnet/results12.png'
# image13 = path + 'resnet/results13.png'
# image14 = path + 'resnet/results14.png'
# image15 = path + 'resnet/results15.png'
# image16 = path + 'resnet/results16.png'
# image17 = path + 'resnet/results17.png'
# image18 = path + 'resnet/results18.png'
# image19 = path + 'resnet/results19.png'
# image20 = path + 'resnet/results20.png'

# unet
# image1  = path + 'unet/results1.png'
# image2  = path + 'unet/results2.png'
# image3  = path + 'unet/results3.png'
# image4  = path + 'unet/results4.png'
# image5  = path + 'unet/results5.png'
# image6  = path + 'unet/results6.png'
# image7  = path + 'unet/results7.png'
# image8  = path + 'unet/results8.png'
# image9  = path + 'unet/results9.png'
# image10 = path + 'unet/results10.png'
# image11 = path + 'unet/results11.png'
# image12 = path + 'unet/results12.png'
# image13 = path + 'unet/results13.png'
# image14 = path + 'unet/results14.png'
# image15 = path + 'unet/results15.png'
# image16 = path + 'unet/results16.png'
# image17 = path + 'unet/results17.png'
# image18 = path + 'unet/results18.png'
# image19 = path + 'unet/results19.png'
# image20 = path + 'unet/results20.png'
# mifcn/mifcn2 results
# image1  = path + 'mifcn/mifcn2/1.tif'
# image2  = path + 'mifcn/mifcn2/2.tif'
# image3  = path + 'mifcn/mifcn2/3.tif'
# image4  = path + 'mifcn/mifcn2/4.tif'
# image5  = path + 'mifcn/mifcn2/5.tif'
# image6  = path + 'mifcn/mifcn2/6.tif'
# image7  = path + 'mifcn/mifcn2/7.tif'
# image8  = path + 'mifcn/mifcn2/8.tif'
# image9  = path + 'mifcn/mifcn2/9.tif'
# image10 = path + 'mifcn/mifcn2/10.tif'
# image11 = path + 'mifcn/mifcn2/11.tif'
# image12 = path + 'mifcn/mifcn2/12.tif'
# image13 = path + 'mifcn/mifcn2/13.tif'
# image14 = path + 'mifcn/mifcn2/14.tif'
# image15 = path + 'mifcn/mifcn2/15.tif'
# image16 = path + 'mifcn/mifcn2/16.tif'
# image17 = path + 'mifcn/mifcn2/17.tif'
# image18 = path + 'mifcn/mifcn2/18.tif'
# image19 = path + 'mifcn/mifcn2/19.tif'
# image20 = path + 'mifcn/mifcn2/20.tif'

#shared encoder results
# image1  = path + 'SSE1/resized/A2_1_.tif'
# image2  = path + 'SSE1/resized/A2_2_.tif'
# image3  = path + 'SSE1/resized/A2_3_.tif'
# image4  = path + 'SSE1/resized/A2_4_.tif'
# image5  = path + 'SSE1/resized/A2_5_.tif'
# image6  = path + 'SSE1/resized/A2_6_.tif'
# image7  = path + 'SSE1/resized/A2_7_.tif'
# image8  = path + 'SSE1/resized/A2_8_.tif'
# image9  = path + 'SSE1/resized/A2_9_.tif'
# image10 = path + 'SSE1/resized/A2_10_.tif'
# image11 = path + 'SSE1/resized/A2_11_.tif'
# image12 = path + 'SSE1/resized/A2_12_.tif'
# image13 = path + 'SSE1/resized/A2_13_.tif'
# image14 = path + 'SSE1/resized/A2_14_.tif'
# image15 = path + 'SSE1/resized/A2_15_.tif'
# image16 = path + 'SSE1/resized/A2_16_.tif'
# image17 = path + 'SSE1/resized/A2_17_.tif'
# image18 = path + 'SSE1/resized/A2_18_.tif'
# image19 = path + 'SSE1/resized/A2_19_.tif'
# image20 = path + 'SSE1/resized/A2_20_.tif'

# truth_image1  = path + 'average1.png'
# truth_image2  = path + 'average2.png'
# truth_image3  = path + 'average3.png'
# truth_image4  = path + 'average4.png'
# truth_image5  = path + 'average5.png'
# truth_image6  = path + 'average6.png'
# truth_image7  = path + 'average7.png'
# truth_image8  = path + 'average8.png'
# truth_image9  = path + 'average9.png'
# truth_image10 = path + 'average10.png'
# truth_image11 = path + 'average11.png'
# truth_image12 = path + 'average12.png'
# truth_image13 = path + 'average13.png'
# truth_image14 = path + 'average14.png'
# truth_image15 = path + 'average15.png'
# truth_image16 = path + 'average16.png'
# truth_image17 = path + 'average17.png'
# truth_image18 = path + 'average18.png'

input_image1  = path + 'raw1.png'
input_image2  = path + 'raw2.png'
input_image3  = path + 'raw3.png'
input_image4  = path + 'raw4.png'
input_image5  = path + 'raw5.png'
input_image6  = path + 'raw6.png'
input_image7  = path + 'raw7.png'
input_image8  = path + 'raw8.png'
input_image9  = path + 'raw9.png'
input_image10 = path + 'raw10.png'
input_image11 = path + 'raw11.png'
input_image12 = path + 'raw12.png'
input_image13 = path + 'raw13.png'
input_image14 = path + 'raw14.png'
input_image15 = path + 'raw15.png'
input_image16 = path + 'raw16.png'
input_image17 = path + 'raw17.png'
input_image18 = path + 'raw18.png'
input_image19 = path + 'raw19.png'
input_image20 = path + 'raw20.png'



def std_mean_back(a): 
    a = np.asanyarray(a) 
    m = np.mean(a) 
    sd = np.std(a) 
    return m, sd

def std_mean_fore(a, b, c): 
    a = np.asanyarray(a) 
    b = np.asanyarray(b) 
    c = np.asanyarray(c) 
    d = np.append(a, b)
    d = np.append(d, c)
    m = np.mean(d) 
    sd = np.std(d) 
    return m, sd

def msr_cnr(im, box1, box2, box3, box4):
	im_back=im.crop(box1)
	im_fore1=im.crop(box2)
	im_fore2=im.crop(box3)
	im_fore3=im.crop(box4)

	# im_fore2.show()
	# print(im_fore2.size)
	back_image = img_to_array(im_back)
	fore1_image = img_to_array(im_fore1)
	fore2_image = img_to_array(im_fore2)
	fore3_image = img_to_array(im_fore3)

	mean_back, std_back = std_mean_back(back_image[:,:,0])
	print(mean_back, std_back)
	mean_fore, std_fore = std_mean_fore(fore1_image[:,:,0], fore1_image[:,:,0], fore1_image[:,:,0])
	print(mean_fore, std_fore)
	return (mean_fore/std_fore), (mean_fore-mean_back) / np.sqrt(0.5 * ((std_back*std_back) + (std_fore*std_fore)) )  


def corr(a, b):
	return np.sum(np.multiply(a, b))

def texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4):
	a=im.crop(box1)
	b=im.crop(box2)
	c=im.crop(box3)
	d=im.crop(box4)
	
	a1=input_im.crop(box1)
	b1=input_im.crop(box2)
	c1=input_im.crop(box3)
	d1=input_im.crop(box4)

	m1, std1 = std_mean_back(a);
	m2, std2 = std_mean_back(b);
	m3, std3 = std_mean_back(c);
	m4, std4 = std_mean_back(d);
	m11, std11 = std_mean_back(a1);
	m12, std12 = std_mean_back(b1);
	m13, std13 = std_mean_back(c1);
	m14, std14 = std_mean_back(d1);
	tp1 = (std1/std11) * np.sqrt((m1/m11))
	tp2 = (std2/std12) * np.sqrt((m1/m12))
	tp3 = (std3/std13) * np.sqrt((m1/m13))
	tp4 = (std4/std14) * np.sqrt((m1/m14))

	l1 = cv.Laplacian(np.asarray(a), ddepth, ksize=kernel_size)
	l2 = cv.Laplacian(np.asarray(b), ddepth, ksize=kernel_size)
	l3 = cv.Laplacian(np.asarray(c), ddepth, ksize=kernel_size)
	l4 = cv.Laplacian(np.asarray(d), ddepth, ksize=kernel_size)

	l1 = l1 - np.mean(l1)
	l2 = l2 - np.mean(l2)
	l3 = l3 - np.mean(l3)
	l4 = l4 - np.mean(l4)

	l11 = cv.Laplacian(np.asarray(a1), ddepth, ksize=kernel_size)
	l21 = cv.Laplacian(np.asarray(b1), ddepth, ksize=kernel_size)
	l31 = cv.Laplacian(np.asarray(c1), ddepth, ksize=kernel_size)
	l41 = cv.Laplacian(np.asarray(d1), ddepth, ksize=kernel_size)

	l11 = l11 - np.mean(l11)
	l21 = l21 - np.mean(l21)
	l31 = l31 - np.mean(l31)
	l41 = l41 - np.mean(l41)
	
	ep1 = corr(l11, l1)/np.sqrt(corr(l11, l11) * corr(l1, l1))
	ep2 = corr(l21, l2)/np.sqrt(corr(l21, l21) * corr(l2, l2))
	ep3 = corr(l31, l3)/np.sqrt(corr(l31, l31) * corr(l3, l3))
	ep4 = corr(l41, l4)/np.sqrt(corr(l41, l41) * corr(l4, l4))

	return (tp2+tp3+tp4)/3, (m1 * m1)/(std1*std1), (ep2+ep3+ep4)/3



msr = []
cnr = []
tp = []
enl = []
ep = []

im=Image.open(image1)
# im.show()
print(im.size)
# Background
box1=(375,5,500,30)

# Left foreground
box2=(100,100,220,150)

# Middle foreground
box3=(375,120,500,170)

# Right foreground
box4=(715,110,835,160)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image1)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)




im=Image.open(image2)
# im.show()
print(im.size)
# Background
box1=(375,10,545,30)

# Left foreground
box2=(100,90,220,140)

# Middle foreground
box3=(400,105,520,155)

# Right foreground
box4=(700,95,820,145)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image2)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image3)
# im.show()
print(im.size)
# Background
box1=(420,5,520,20)

# Left foreground
box2=(100,80,220,130)

# Middle foreground
box3=(400,105,520,155)

# Right foreground
box4=(700,95,820,145)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image3)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image4)
# im.show()
print(im.size)
# Background
box1=(375,10,455,35)

# Left foreground
box2=(100,110,205,160)

# Middle foreground
box3=(400,135,520,185)

# Right foreground
box4=(675,130,795,180)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image4)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image5)
# im.show()
print(im.size)
# Background
box1=(425,20,545,55)

# Left foreground
box2=(120,120,240,170)

# Middle foreground
box3=(400,135,520,185)

# Right foreground
box4=(715,135,835,185)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image5)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image6)
# im.show()
print(im.size)
# Background
box1=(410,5,520,25)

# Left foreground
box2=(100,105,220,155)

# Middle foreground
box3=(385,120,505,170)

# Right foreground
box4=(670,115,790,165)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image6)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image7)
# im.show()
print(im.size)
# Background
box1=(400,15,500,40)

# Left foreground
box2=(60,120,180,170)

# Middle foreground
box3=(390,140,510,190)

# Right foreground
box4=(700,155,820,205)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image7)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image8)
# im.show()
print(im.size)
# Background
box1=(440,15,530,40)

# Left foreground
box2=(105,110,225,160)

# Middle foreground
box3=(380,125,500,175)

# Right foreground
box4=(675,145,795,195)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image8)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image9)
# im.show()
print(im.size)
# Background
box1=(490,5,575,30)

# Left foreground
box2=(100,105,220,155)

# Middle foreground
box3=(385,125,505,175)

# Right foreground
box4=(670,130,790,180)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image9)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image10)
# im.show()
print(im.size)
# Background
box1=(520,10,620,40)

# Left foreground
box2=(110,105,230,155)

# Middle foreground
box3=(395,140,515,190)

# Right foreground
box4=(675,140,795,190)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image10)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image11)
# im.show()
print(im.size)
# Background
box1=(405,20,530,50)

# Left foreground
box2=(80,110,200,160)

# Middle foreground
box3=(390,145,510,195)

# Right foreground
box4=(695,150,815,200)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image11)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image12)
# im.show()
print(im.size)
# Background
box1=(450,25,550,60)

# Left foreground
box2=(80,145,200,195)

# Middle foreground
box3=(400,165,520,205)

# Right foreground
box4=(730,160,850,210)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image12)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image13)
# im.show()
print(im.size)
# Background
box1=(385,10,470,30)

# Left foreground
box2=(80,115,200,165)

# Middle foreground
box3=(365,110,485,160)

# Right foreground
box4=(690,105,810,155)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image13)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image14)
# im.show()
print(im.size)
# Background
box1=(430,5,485,20)

# Left foreground
box2=(100,80,220,130)

# Middle foreground
box3=(385,80,505,130)

# Right foreground
box4=(695,85,815,135)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image14)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)



im=Image.open(image15)
# im.show()
print(im.size)
# Background
box1=(730,5,800,15)

# Left foreground
box2=(100,80,220,130)

# Middle foreground
box3=(400,80,520,130)

# Right foreground
box4=(700,85,820,135)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image15)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image16)
# im.show()
print(im.size)
# Background
box1=(695,10,790,30)

# Left foreground
box2=(135,95,255,145)

# Middle foreground
box3=(350,95,470,145)

# Right foreground
box4=(710,100,830,150)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image16)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image17)
# im.show()
print(im.size)
# Background
box1=(390,20,470,45)

# Left foreground
box2=(115,135,235,185)

# Middle foreground
box3=(380,120,500,170)

# Right foreground
box4=(715,130,835,180)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image17)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image18)
# im.show()
print(im.size)
# Background
box1=(395,10,495,35)

# Left foreground
box2=(100,45,220,95)

# Middle foreground
box3=(380,145,500,195)

# Right foreground
box4=(660,145,780,195)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image18)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image19)
# im.show()
print(im.size)
# Background
box1=(350,20,430,50)

# Left foreground
box2=(120,165,240,205)

# Middle foreground
box3=(360,170,480,220)

# Right foreground
box4=(655,140,775,190)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image19)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)


im=Image.open(image20)
# im.show()
print(im.size)
# Background
box1=(420,15,500,40)

# Left foreground
box2=(145,115,265,165)

# Middle foreground
box3=(400,140,520,190)

# Right foreground
box4=(680,120,800,170)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image20)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)

print(msr)
print(cnr)
print(tp)
print(enl)
print(ep)
print("MSR = " + str(np.array(msr).mean()) + "\n")
print("CNR = " + str(np.array(cnr).mean()) + "\n")
print("tp = " + str(np.array(tp).mean()) + "\n")
print("ENL = " + str(np.array(enl).mean()) + "\n")
print("EP = " + str(np.array(ep).mean()) + "\n")