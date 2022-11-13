# ATRGAN
The pytorch implemtation of the paper "Asymmetric Training in RealnessGAN".

# Implementation
**Requirement**
* pytorch (1.3-1.5 version)
* python 3.7.3
* tensorflow (lastest version, don't need to download if you don't need to calculcate fid)
**Before you run the code**
* make sure all required folders are created. (including an output folder to save model, an extra folder to save generated images and an inception folder for inception model.)
* if you want to use [Cat Dataset](http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd): Run setting_up_script.sh in same folder as preprocess_cat_dataset.py and your CAT dataset (open and run manually)
**Usage**
* run `python train.py` (please check the options.py carefully.)
* notes: similar to RealnessGAN, the random seed for training input is 1 constantly. It is important to notice that, although the random seedes are same, but the trained result will still be various.
* run `python train.py --num_outcomes=50 --image_size=256 --G_h_size=32 --D_h_size=32 --total_iters=350000` for CelebA-HQ-256x256 [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans), run `python train.py --num_outcomes=8 --image_size=64 --G_h_size=64 --D_h_size=64 --total_iters=400000` for CAT-64x64.
**to calculate the FID sorce**
* make sure you save the generated images in the extra folder for calculation
* run `python fid.py "/path/to/saved_generated_image/dir/" "/path/to/real_image/dir" -i "/path/to/Inception/dir" --gpu "0"` (you can also run `python fid_H.py "/path/to/saved_generated_image/dir/" "/path/to/real_image/dir" -i "/path/to/Inception/dir" --gpu "0"` with lower video memory.)

# Snapshots

**CelebA-HQ 256x256 (FID = 20.47)**

![](/images/ATRGAN.png)
