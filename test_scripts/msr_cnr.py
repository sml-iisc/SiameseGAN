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
# path = cwd + '/results_pre/'
path = cwd + '/Results/dataset1/'
# path = '/home/skryzor/workspace/image_denoise/dataset2/results/'

# path = "/home/nilesh2019/workspace/image_denoise/GAN/Results/"

# msssim
# image1  = path + 'results1.png'
# image2  = path + 'results2.png'
# image3  = path + 'results3.png'
# image4  = path + 'results4.png'
# image5  = path + 'results5.png'
# image6  = path + 'results6.png'
# image7  = path + 'results7.png'
# image8  = path + 'results8.png'
# image9  = path + 'results9.png'
# image10 = path + 'results10.png'
# image11 = path + 'results11.png'
# image12 = path + 'results12.png'
# image13 = path + 'results13.png'
# image14 = path + 'results14.png'
# image15 = path + 'results15.png'
# image16 = path + 'results16.png'
# image17 = path + 'results17.png'
# image18 = path + 'results18.png'

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


# Ideal results
# image1  = path + 'average1.png'
# image2  = path + 'average2.png'
# image3  = path + 'average3.png'
# image4  = path + 'average4.png'
# image5  = path + 'average5.png'
# image6  = path + 'average6.png'
# image7  = path + 'average7.png'
# image8  = path + 'average8.png'
# image9  = path + 'average9.png'
# image10 = path + 'average10.png'
# image11 = path + 'average11.png'
# image12 = path + 'average12.png'
# image13 = path + 'average13.png'
# image14 = path + 'average14.png'
# image15 = path + 'average15.png'
# image16 = path + 'average16.png'
# image17 = path + 'average17.png'
# image18 = path + 'average18.png'


# mifcn results
# image1  = path + 'mifcn/1.tif'
# image2  = path + 'mifcn/2.tif'
# image3  = path + 'mifcn/3.tif'
# image4  = path + 'mifcn/4.tif'
# image5  = path + 'mifcn/5.tif'
# image6  = path + 'mifcn/6.tif'
# image7  = path + 'mifcn/7.tif'
# image8  = path + 'mifcn/8.tif'
# image9  = path + 'mifcn/9.tif'
# image10 = path + 'mifcn/10.tif'
# image11 = path + 'mifcn/11.tif'
# image12 = path + 'mifcn/12.tif'
# image13 = path + 'mifcn/13.tif'
# image14 = path + 'mifcn/14.tif'
# image15 = path + 'mifcn/15.tif'
# image16 = path + 'mifcn/16.tif'
# image17 = path + 'mifcn/17.tif'
# image18 = path + 'mifcn/18.tif'

# shared encoder results
# image1  = path + 'SSE1/A2_raw1_.png'
# image2  = path + 'SSE1/A2_raw2_.png'
# image3  = path + 'SSE1/A2_raw3_.png'
# image4  = path + 'SSE1/A2_raw4_.png'
# image5  = path + 'SSE1/A2_raw5_.png'
# image6  = path + 'SSE1/A2_raw6_.png'
# image7  = path + 'SSE1/A2_raw7_.png'
# image8  = path + 'SSE1/A2_raw8_.png'
# image9  = path + 'SSE1/A2_raw9_.png'
# image10 = path + 'SSE1/A2_raw10_.png'
# image11 = path + 'SSE1/A2_raw11_.png'
# image12 = path + 'SSE1/A2_raw12_.png'
# image13 = path + 'SSE1/A2_raw13_.png'
# image14 = path + 'SSE1/A2_raw14_.png'
# image15 = path + 'SSE1/A2_raw15_.png'
# image16 = path + 'SSE1/A2_raw16_.png'
# image17 = path + 'SSE1/A2_raw17_.png'
# image18 = path + 'SSE1/A2_raw18_.png'

truth_image1  = path + 'average1.png'
truth_image2  = path + 'average2.png'
truth_image3  = path + 'average3.png'
truth_image4  = path + 'average4.png'
truth_image5  = path + 'average5.png'
truth_image6  = path + 'average6.png'
truth_image7  = path + 'average7.png'
truth_image8  = path + 'average8.png'
truth_image9  = path + 'average9.png'
truth_image10 = path + 'average10.png'
truth_image11 = path + 'average11.png'
truth_image12 = path + 'average12.png'
truth_image13 = path + 'average13.png'
truth_image14 = path + 'average14.png'
truth_image15 = path + 'average15.png'
truth_image16 = path + 'average16.png'
truth_image17 = path + 'average17.png'
truth_image18 = path + 'average18.png'

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


im_width = 896
im_height = 448
border = 5
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.transform import resize
import imageio

def get_image(image_path):
	output = np.zeros((1, im_height, im_width, 1), dtype=np.float32)
	img = load_img(image_path, grayscale=True)
	x_img = img_to_array(img)
	x_img = resize(x_img, (448,896, 1), mode='constant', preserve_range=True)
	output[0, ..., 0] = x_img.squeeze() / 255
	output = output[0, :, :, :]
	return output[:,:,0]
	# return imageio.imread(image_path)


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

	m_im, std_im = std_mean_back(im) 
	m_input_im, std_input_im = std_mean_back(input_im) 

	tp1 = (std1/std11) * np.sqrt((m_im/m_input_im))
	tp2 = (std2/std12) * np.sqrt((m_im/m_input_im))
	tp3 = (std3/std13) * np.sqrt((m_im/m_input_im))
	tp4 = (std4/std14) * np.sqrt((m_im/m_input_im))

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

	return (tp2+tp3+tp4)/3, (m1 * m1)/(std1 * std1), (ep2+ep3+ep4)/3



def msr_cnr(im, box1, box2, box3, box4):
	im_back=im.crop(box1)
	im_fore1=im.crop(box2)
	im_fore2=im.crop(box3)
	im_fore3=im.crop(box4)

	# im_back.show()
	# im_fore1.show()
	# im_fore2.show()
	# im_fore3.show()
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

ssim = []
psnr = []
msr = []
cnr = []
tp = []
enl = []
ep = []
im=Image.open(image1)
# im.show()
print(im.size)
# Background
box1=(400,15,495,45)

# Left foreground
box2=(75,100,195,150)

# Middle foreground
box3=(400,110,520,170)

# Right foreground
box4=(700,110,820,170)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image1)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image1)
out_im = get_image(image1)

im=Image.open(image2)
# im.show()
print(im.size)
# Background
box1=(400,15,515,45)

# Left foreground
# box2=(75,80,190,130)
# 
# Middle foreground
box3=(410,100,540,150)

# Right foreground
box4=(685,100,805,150)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image2)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image2)
out_im = get_image(image2)



im=Image.open(image3)
# im.show()
print(im.size)
# Background
box1=(400,40,500,85)
# 
Left foreground
box2=(80,280,200,330)

# Middle foreground
box3=(400,245,545,295)

# Right foreground
box4=(710,210,830,265)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image3)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image3)
out_im = get_image(image3)

im=Image.open(image4)
# im.show()
print(im.size)
# Background
box1=(390,5,495,30)

# Left foreground
# box2=(80,75,210,130)
# 
# Middle foreground
box3=(375,90,535,140)

# Right foreground
box4=(675,85,800,135)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image4)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image4)
out_im = get_image(image4)




im=Image.open(image5)
# im.show()
print(im.size)
# Background
# box1=(395,10,515,45)
# 
# Left foreground
box2=(395,100,515,155)

# Middle foreground
box3=(100,100,220,155)

# Right foreground
box4=(700,105,820,155)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image5)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image5)
out_im = get_image(image5)



im=Image.open(image6)
# im.show()
print(im.size)
# Background
box1=(350,10,465,40)
# 
Left foreground
box2=(70,100,200,150)

# Middle foreground
box3=(340,110,470,170)

# Right foreground
box4=(665,90,800,140)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image6)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image6)
out_im = get_image(image6)







im=Image.open(image7)
im.show()
# print(im.size)
# Background
box1=(450,10,490,45)

# Left foreground
box2=(95,80,250,140)

# Middle foreground
box3=(375,70,550,145)

# Right foreground
box4=(700,110,835,175)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image7)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image7)
out_im = get_image(image7)



im=Image.open(image8)
# im.show()
print(im.size)
# Background
box1=(400,15,515,50)
# 
Left foreground
box2=(115,135,255,185)

# Middle foreground
box3=(415,110,600,150)

# Right foreground
box4=(690,95,810,135)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image8)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image8)
out_im = get_image(image8)



im=Image.open(image9)
# im.show()
print(im.size)
# Background
box1=(410,15,510,45)
# 
Left foreground
box2=(80,130,200,180)

# Middle foreground
box3=(385,120,525,170)

# Right foreground
box4=(640,130,760,180)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image9)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image9)
out_im = get_image(image9)


im=Image.open(image10)
# im.show()
print(im.size)
# Background
box1=(405,5,480,30)

Left foreground
# box2=(105,85,235,140)

# Middle foreground
box3=(380,85,410,145)

# Right foreground
box4=(635,65,755,120)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image10)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image10)
out_im = get_image(image10)



im=Image.open(image11)
# im.show()
print(im.size)
# Background
box1=(435,20,520,55)
# 
Left foreground
box2=(85,75,205,135)

# Middle foreground
box3=(385,135,520,185)

# Right foreground
box4=(685,135,805,185)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image11)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image11)
out_im = get_image(image11)



im=Image.open(image12)
# im.show()
print(im.size)
# Background
box1=(410,10,500,45)
# 
Left foreground
box2=(70,90,190,140)

# Middle foreground
box3=(385,90,505,140)

# Right foreground
box4=(600,90,720,140)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image12)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image12)
out_im = get_image(image12)



im=Image.open(image13)
# im.show()
print(im.size)
# Background
box1=(400,5,450,25)
# 
Left foreground
box2=(100,80,220,130)

# Middle foreground
box3=(375,75,520,140)

# Right foreground
box4=(700,70,820,120)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image13)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image13)
out_im = get_image(image13)


im=Image.open(image14)
# im.show()
print(im.size)
# Background
box1=(385,5,470,30)

Left foreground
# box2=(90,80,210,130)

# Middle foreground
box3=(380,90,400,140)

# Right foreground
box4=(660,80,780,130)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image14)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image14)
out_im = get_image(image14)


im=Image.open(image15)
# im.show()
print(im.size)
# Background
box1=(360,20,440,50)

Left foreground
# box2=(60,160,180,210)

# Middle foreground
box3=(360,185,490,240)

# Right foreground
box4=(735,160,850,210)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image15)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image15)
out_im = get_image(image15)



im=Image.open(image16)
# im.show()
print(im.size)
# Background
box1=(415,10,500,40)
# 
Left foreground
box2=(90,110,220,160)

# Middle foreground
box3=(395,100,520,150)

# Right foreground
box4=(695,110,820,160)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image16)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image16)
out_im = get_image(image16)


im=Image.open(image17)
# im.show()
print(im.size)
# Background
box1=(415,15,505,50)

Left foreground
# box2=(90,110,210,160)

# Middle foreground
box3=(395,115,520,165)

# Right foreground
box4=(660,110,780,160)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image17)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image17)
out_im = get_image(image17)



im=Image.open(image18)
# im.show()
print(im.size)
# Background
box1=(420,15,505,45)
# 
Left foreground
box2=(100,115,220,165)

# Middle foreground
box3=(370,115,500,170)

# Right foreground
box4=(655,120,780,170)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)
input_im=Image.open(input_image18)
t, e, e2 = texture_and_edge_preservation_equvi_no_of_looks(im, input_im, box1, box2, box3, box4)
tp.append(t)
enl.append(e)
ep.append(e2)
t_image = get_image(truth_image18)
out_im = get_image(image18)

print(msr)
print(cnr)
print(tp)
print(enl)
print(ep)

print("MSR = " + str(np.array(msr).mean()) + "\n")
print("CNR = " + str(np.array(cnr).mean()) + "\n")
print("tp = " + str(np.array(tp).mean()) + "\n")
print("EP = " + str(np.array(ep).mean()) + "\n")
