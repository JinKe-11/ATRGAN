# ATRGAN
The pytorch implemtation of the paper "Asymmetric Training in RealnessGAN".

# Implementation
**Requirement**
* pytorch (1.3-1.5 version)
* python 3.7.3
* tensorflow (lastest version, don't need to download if you don't need to calculcate fid)
**Before you run the code**
* make sure all required folders are created, including a output folder to save model, an extra folder to save generated images and an inception folder for inception model. 

* if you want to use [Cat Dataset](http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd): Run setting_up_script.sh in same folder as preprocess_cat_dataset.py and your CAT dataset (open and run manually)

**Usage**
* run 'GAN_losses_iter.py' (please check the arguements in the code carefully, including the argument for change the model and hyperprarmeter)

* e.g. to train a PUSGAN model with 64x64 size images: `python3 GAN_losses_iter.py --image_size=64 --loss_D=5 --lr_D .0002 --lr_G .0002 --batch_size 32 --Diters 1 --prior=0.3 --prior_increase_mode=1 --seed=1 --input_folder=/path/to/input_image/dir/ --output_folder=/path/to/output/dir --extra_folder=/path/to/generated_image/dir --inception_folder=/path/to/inception/dir`

* notes: similar to RelativisticGAN, the random seed for training input is 1 constantly. It is important to notice that, although the random seedes are same, but the trained result will still be various.

**to calculate the FID sorce**
* make sure you save the generated images in the extra folder for calculation
* run `python fid.py "/path/to/saved_generated_image/dir/01" "/path/to/real_image/dir" -i "/path/to/Inception/dir" --gpu "0"`
