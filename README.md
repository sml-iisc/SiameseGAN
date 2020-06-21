## Requirements

* tensorflow 1.13.1
* keras 2.2.4
* tqdm 5.0.0
* ipywidgets
* scipy 0.17.0
* scikit-image
* scikit-learn
* matplotlib
* pandas

# The existing code uses the SDOCT dataset containing 28 images with 10 images for training and 18 for testing purpose.
Dataset is present in "dataset/train" and "dataset/test" directories.
Dataset 2 images are present in the "dataset/test/real" directory. 

# The training and testing can be done by running the following scripts
* cd /path_to_the_project/train_scripts

* training proposed model SiameseGAN - "python GAN-SIAMESE.py"
* training WGAN - "python GAN-ResNet.py"
* training SiameseGAN with UNET generator - "python GAN-SIAMESE-UNET.py"


* cd /path_to_the_project/test_scripts

* testing with ResNet generator- "python GAN-test-Siamese.py"
* testing with UNET generator- "python GAN-test_Siamese_UNET.py"

By default the models are trained with combined perceptual and MSSSIM loss

# Calculate MSR and CNR values for dataset-1 and dataset-2 using this scripts
* Dataset-1 - "python msr_cnr.py"
* Dataset-2 - "python msr_cnr2.py"


### All the models are defined in file "Models/models.py"

### All the parameters of the network once trained will be saved in directory "GAN/" and the results will be saved in the "GAN/results.txt" and output images will be saved in "GAN/Results/"

### The pretrained model is provided in the directory as "saved_model/generator.h5" (SiameseGAN (MS-SSIM) model) and gives the best results
* other models are also saved in the same folder (WGAN-ResNet)

### The directory "other_scripts/" contain the notebook files and other python files which are not the part of the main code

### To remove noise from images using our best model, run following script
* Go to the test script folder
* images_directory_path = absolute path to the directory of images which needs to be denoised eg. lets say we want to denoise dataset 2 images present in real directory
* "python image_denoise /home/user/SiameseGAN-master/dataset/test/real/"


### Hardware used 
* 12 GB GPU NVIDIA GeForce 1080 Ti
* 32 GB RAM

### Reference Work
Links for the models which are used for comaparison.

MSBTD - http://people.duke.edu/~sf59/software.html
MIFCN - https://github.com/ashkan-abbasi66/MIFCN
Shared Encoder - https://github.com/adigasu/SE_Despeckling