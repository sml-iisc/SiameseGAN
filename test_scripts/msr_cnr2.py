from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
os.chdir("../")
cwd = os.getcwd()
path = cwd + '/dataset/test/real/Results/'
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


msr = []
cnr = []

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


print(msr)
print(cnr)
print("MSR = " + str(np.array(msr).mean()) + "\n")
print("CNR = " + str(np.array(cnr).mean()) + "\n")
for i in range(len(msr)):
	print(msr[i], cnr[i])
exit(0)