## Requirements

* keras 2.2.4
* tensorflow 1.13

#The existing code uses the SDOCT dataset containing 28 images with 10 images for training and 18 for testing purpose.
Dataset is present in "dataset/train" and "dataset/test" directories

# The training and testing can be done by running the following scripts

* training proposed model SiameseGAN - "train_scripts/python GAN-SIAMESE.py"
* training WGAN - "train_scripts/python GAN-ResNet.py"
* training SiameseGAN with UNET generator - "train_scripts/python GAN-SIAMESE-UNET.py"


* testing with ResNet generator- "test_scripts/python GAN-test-Siamese.py"
* testing with UNET generator- "test_scripts/python GAN-test_Siamese_UNET.py"
By default the models are trained with combined perceptual and MSSSIM loss

### All the models are defined in file "Models/models.py"


### All the parameters of the network once trained will be saved in directory "GAN/" and the results will be saved in the "GAN/results.txt" and output images will be saved in "GAN/Results/"

### The pretrained model is provided in the directory as "GAN/generator.h5" and gives the best results

### The directory "other_scripts/" contain the notebook files and other python files which are not the part of the main code
