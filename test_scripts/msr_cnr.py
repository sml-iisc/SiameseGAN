from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

os.chdir("../")
cwd = os.getcwd()
path = cwd + '/Results/'
# path = '/home/skryzor/workspace/image_denoise/dataset2/results/'

# path = "/home/nilesh2019/workspace/image_denoise/GAN/Results/"


image1  = path + 'results1.png'
image2  = path + 'results2.png'
image3  = path + 'results3.png'
image4  = path + 'results4.png'
image5  = path + 'results5.png'
image6  = path + 'results6.png'
image7  = path + 'results7.png'
image8  = path + 'results8.png'
image9  = path + 'results9.png'
image10 = path + 'results10.png'
image11 = path + 'results11.png'
image12 = path + 'results12.png'
image13 = path + 'results13.png'
image14 = path + 'results14.png'
image15 = path + 'results15.png'
image16 = path + 'results16.png'
image17 = path + 'results17.png'
image18 = path + 'results18.png'


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

msr = []
cnr = []

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


im=Image.open(image2)
# im.show()
print(im.size)
# Background
box1=(400,15,515,45)

# Left foreground
box2=(75,80,190,130)

# Middle foreground
box3=(410,100,540,150)

# Right foreground
box4=(685,100,805,150)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)



im=Image.open(image3)
# im.show()
print(im.size)
# Background
box1=(400,40,500,85)

# Left foreground
box2=(80,280,200,330)

# Middle foreground
box3=(400,245,545,295)

# Right foreground
box4=(710,210,830,265)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)


im=Image.open(image4)
# im.show()
print(im.size)
# Background
box1=(390,5,495,30)

# Left foreground
box2=(80,75,210,130)

# Middle foreground
box3=(375,90,535,140)

# Right foreground
box4=(675,85,800,135)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)





im=Image.open(image5)
# im.show()
print(im.size)
# Background
box1=(395,10,515,45)

# Left foreground
box2=(395,100,515,155)

# Middle foreground
box3=(100,100,220,155)

# Right foreground
box4=(700,105,820,155)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)



im=Image.open(image6)
# im.show()
print(im.size)
# Background
box1=(350,10,465,40)

# Left foreground
box2=(70,100,200,150)

# Middle foreground
box3=(340,110,470,170)

# Right foreground
box4=(665,90,800,140)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)








im=Image.open(image7)
# im.show()
print(im.size)
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




im=Image.open(image8)
# im.show()
print(im.size)
# Background
box1=(400,15,515,50)

# Left foreground
box2=(115,135,255,185)

# Middle foreground
box3=(415,110,600,150)

# Right foreground
box4=(690,95,810,135)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)




im=Image.open(image9)
# im.show()
print(im.size)
# Background
box1=(410,15,510,45)

# Left foreground
box2=(80,130,200,180)

# Middle foreground
box3=(385,120,525,170)

# Right foreground
box4=(640,130,760,180)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)



im=Image.open(image10)
# im.show()
print(im.size)
# Background
box1=(405,5,480,30)

# Left foreground
box2=(105,85,235,140)

# Middle foreground
box3=(380,85,410,145)

# Right foreground
box4=(635,65,755,120)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)



im=Image.open(image11)
# im.show()
print(im.size)
# Background
box1=(435,20,520,55)

# Left foreground
box2=(85,75,205,135)

# Middle foreground
box3=(385,135,520,185)

# Right foreground
box4=(685,135,805,185)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)


im=Image.open(image12)
# im.show()
print(im.size)
# Background
box1=(410,10,500,45)

# Left foreground
box2=(70,90,190,140)

# Middle foreground
box3=(385,90,505,140)

# Right foreground
box4=(600,90,720,140)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)



im=Image.open(image13)
# im.show()
print(im.size)
# Background
box1=(400,5,450,25)

# Left foreground
box2=(100,80,220,130)

# Middle foreground
box3=(375,75,520,140)

# Right foreground
box4=(700,70,820,120)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)



im=Image.open(image14)
# im.show()
print(im.size)
# Background
box1=(385,5,470,30)

# Left foreground
box2=(90,80,210,130)

# Middle foreground
box3=(380,90,400,140)

# Right foreground
box4=(660,80,780,130)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)



im=Image.open(image15)
# im.show()
print(im.size)
# Background
box1=(360,20,440,50)

# Left foreground
box2=(60,160,180,210)

# Middle foreground
box3=(360,185,490,240)

# Right foreground
box4=(735,160,850,210)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)



im=Image.open(image16)
# im.show()
print(im.size)
# Background
box1=(415,10,500,40)

# Left foreground
box2=(90,110,220,160)

# Middle foreground
box3=(395,100,520,150)

# Right foreground
box4=(695,110,820,160)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)



im=Image.open(image17)
# im.show()
print(im.size)
# Background
box1=(415,15,505,50)

# Left foreground
box2=(90,110,210,160)

# Middle foreground
box3=(395,115,520,165)

# Right foreground
box4=(660,110,780,160)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)




im=Image.open(image18)
# im.show()
print(im.size)
# Background
box1=(420,15,505,45)

# Left foreground
box2=(100,115,220,165)

# Middle foreground
box3=(370,115,500,170)

# Right foreground
box4=(655,120,780,170)

m, c = msr_cnr(im, box1, box2, box3, box4)
msr.append(m)
cnr.append(c)


print(msr)
print(cnr)
print("MSR = " + str(np.array(msr).mean()) + "\n")
print("CNR = " + str(np.array(cnr).mean()) + "\n")
