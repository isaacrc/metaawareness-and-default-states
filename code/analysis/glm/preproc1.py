#!/usr/bin/env python
# coding: utf-8

# # Preprocess fMRI data  

# This script further preprocesses fmriprep's preprocessed data. It really doesn't do much, and is just here for fun
# specifically, it organizes runs output from fmriprep into lists. Yay!

# ## py conversion

# In[35]:


#jupyter nbconvert --to python slurm_create-data_preproc.ipynb


# ## Imports 

# In[1]:


import warnings
import sys  
import random
# import logging

import deepdish as dd
import numpy as np

import brainiak.eventseg.event
import nibabel as nib
import nilearn as nil
# Import a function from BrainIAK to simulate fMRI data
import brainiak.utils.fmrisim as sim  

from nilearn.input_data import NiftiMasker

import scipy.io
from scipy import stats
from scipy.stats import norm, zscore, pearsonr
from scipy.signal import gaussian, convolve

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import seaborn as sns 



from brainiak import image, io
from scipy.stats import stats
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from nilearn import datasets, plotting
from nilearn.input_data import NiftiSpheresMasker

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, resample_img, mean_img,index_img
from nilearn import image
from nilearn import masking
from nilearn.plotting import view_img
from nilearn.image import resample_to_img, concat_imgs

from nilearn.image import concat_imgs, resample_img, mean_img
from nilearn.plotting import view_img

import numpy as np 
import os
import os.path
import scipy.io
import nibabel as nib
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask, compute_brain_mask
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit
from copy import deepcopy
import brainiak.utils.fmrisim as sim  



# In[2]:


random.seed(10)


# ## custom helper functions 


from utils_anal import load_epi_data, load_conf_data, load_epi_sub032, load_conf_sub032

##########
# ## directories 

# In[41]:


top_dir = '../../..'
data_dir = top_dir + "/data"
work_dir = data_dir + '/work'
mask_dir = work_dir + '/masks'
behav_dir = top_dir + '/data/behavioral'
rois_dir = data_dir + "/rois"
fmri_prep = '/jukebox/graziano/coolCatIsaac/MEI/data/bids/derivatives/fmriprep' ### DOWNLOAD FMRI DATA HERE
conf_dir = work_dir + '/confs'
preproc_dir = work_dir + '/glm_preproc'


# Check if the directory exists, and create it if it doesn't
if not os.path.exists(preproc_dir):
    os.makedirs(preproc_dir)
    print(f"Directory created: {preproc_dir}")


##### MASKING #### 

# Set the size (in terms of X, Y, Z) of the volume we want to create
dimensions = np.asarray([78, 93, 65])

# Generate an anatomical image with the size above of brain voxels in gray matter
# This outputs variables for two versions of the image, binary (mask) and probabilistic (template)
mask, template = sim.mask_brain(dimensions, mask_self=False)
epi = load_epi_data(fmri_prep, 'sub-004', 1, 'MNI') # load template data to get affine
mask_3d = nib.Nifti1Image(mask, epi.affine) ##create nifti image


###


### Preprocess function

def preproc_glm_imgs(fmri_prep, sub, num_runs, space, mask):
    # This is based off of 'load_data' function in template
    # Loads all fMRI runs into a 4D NIFTI list #
    """
    purpose: get a cleaned epi 
    inputs:
        - fmri_prep: path
        - morph = T1 or MNI registration?
        - norm_type = by Space or by Time?
    return: a dictionary of runs, preprocessed 
    """
    run_dic = {}
    print("Begin preproc, u dynamic lil windmill!")
    ## preprocess 7 runs ## 
    for run in range(1, num_runs + 1):
        ### subject specific loading ###  
        if sub == 'sub-003' and run == 6:
            ## load the seventh, instead of sixth run -- this is a quirk of this dataset ##
            adjust_run = run + 1
            # Load EPI
            epi = load_epi_data(fmri_prep, sub, adjust_run, space)

        elif sub == 'sub-012' and run >= 5:
            print('ADJUSTING')
            ## load the 6th, 7th run, instead of 5, 6 -- this is a quirk of this dataset ##
            adjust_run = run + 1
            # Load EPI
            epi = load_epi_data(fmri_prep, sub, adjust_run, space)

        elif sub == 'sub-032': 
            # Load EPI
            epi = load_epi_sub032(fmri_prep, sub, run,space)
        elif sub == 'sub-014' and run >=2: 
            ## skip the second run ##
            adjust_run = run + 1
            # Load EPI
            epi = load_epi_data(fmri_prep, sub, adjust_run, space)
        elif sub == 'sub-029' and run >=4: 
            ## do not use runs 3 + 4 -- this sub only has 5 usable runs ##
            adjust_run = run + 1
            # Load EPI
            epi = load_epi_data(fmri_prep, sub, adjust_run, space)
        elif sub == 'sub-015' and run >=5: 
            ## this sub only has 4 usable runs ##
            continue
        else:
            #### Load the epi if regular sub or run ####
            # Load EPI
            epi = load_epi_data(fmri_prep, sub, run,space)

            print(f'run {run} shape: {epi.shape}')
        
        
        
        ## adjust subject 2 to append two extra TRs to the end of shrek -- quirk of dataset ## 
        if run == 1 and sub == 'sub-002':
            len_b = epi.shape[3]
            print(len_b)
            two_imgs = image.concat_imgs([index_img(epi, 1)]*2)
            print(two_imgs.shape)
            epi = image.concat_imgs([epi, two_imgs])
            print(f'ADJUSTED: {epi.shape}')
        
        #### Save Date 
        run_dic[run] = epi
        print(f'finished run {run}')
    print("FINISHED YAY BEAST")
    return run_dic






# ## main vars 

# In[43]:


### sub_list ###
sub_list = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005','sub-006','sub-007','sub-008','sub-009','sub-010', 'sub-011',
    'sub-012','sub-013','sub-014','sub-105','sub-016','sub-017','sub-018','sub-019','sub-020','sub-021',
    'sub-022','sub-023','sub-024','sub-025','sub-026','sub-027','sub-028','sub-029','sub-030','sub-031','sub-032',
    'sub-033','sub-034','sub-035','sub-036','sub-037','sub-038','sub-039','sub-040', 'sub-041'
]

#sub_list = ['sub-032']

## current sublist does not include subject 011, subject 015, 001 -- not sure why 015 is excluded. 11 became 41 i think 

###### LOADING VARS #######
# Number of runs to load 
num_runs = 6
# Registration ust be either T1 or MNI
space = "MNI"# 

## mask image ##
#mask_img = nib.load(mask_dir + "/whole_b_bnk.nii.gz")



# In[57]:


for sub in sub_list:
    sub_dic = preproc_glm_imgs(fmri_prep, sub, num_runs, space, mask_3d)
    out_name = f'/{sub}_raw_4D_2.npy'
    np.save(preproc_dir + out_name, sub_dic)
    


# In[ ]:




