#!/usr/bin/env python
# coding: utf-8

# # Visualize ISC

# This script finds the networks that exhibit a decrease in ISC over time. Specifically, the 'vis' version perfroms sevral different plotting functions

# ## py conversion

# In[103]:


#jupyter nbconvert --to python slurm_create-data_preproc.ipynb


# ## Imports 

# In[104]:


import warnings
import sys  
import random
import os
import os.path

import deepdish as dd
import numpy as np
import pandas as pd

import scipy.io
from scipy import stats
from scipy.stats import stats
from scipy.stats import norm, zscore, pearsonr
from scipy.signal import gaussian, convolve

#plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
import seaborn as sns 
from matplotlib.lines import Line2D

# nil and nib 
import nibabel as nib
import nilearn as nil

from nilearn.input_data import NiftiMasker
from nilearn import datasets, plotting
from nilearn.plotting import plot_roi
from nilearn.input_data import NiftiSpheresMasker
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.image import concat_imgs, resample_img, mean_img,index_img
from nilearn import image
from nilearn import masking
from nilearn.plotting import view_img
from nilearn.image import resample_to_img
from nilearn.image import concat_imgs, resample_img, mean_img
from nilearn.plotting import view_img
from nilearn.input_data import NiftiMasker
from nilearn.masking import compute_epi_mask, compute_brain_mask, unmask
from nilearn.plotting import plot_stat_map

# Brainiak # 
from brainiak import image, io 
import brainiak.utils.fmrisim as sim  
from brainiak import image, io
import brainiak.eventseg.event
from brainiak.isc import (isc, isfc, bootstrap_isc, permutation_isc,
                          timeshift_isc, phaseshift_isc,
                          compute_summary_statistic)
from brainiak.io import load_boolean_mask, load_images
from statsmodels.stats.multitest import multipletests

from brainiak.isc import squareform_isfc




# In[105]:


random.seed(10)


# ## custom helper functions 

# In[106]:


## imports
from utils_anal import load_epi_data, resample_atlas, get_network_labels


# ## directories 

# In[107]:


top_dir = '/jukebox/graziano/coolCatIsaac/mei'
data_dir = top_dir + "/data"
work_dir = data_dir + '/work'
mask_dir = work_dir + '/masks'
behav_dir = top_dir + '/data/behavioral'
rois_dir = data_dir + "/rois"
fmri_prep = data_dir + '/bids/derivatives/fmriprep'
conf_dir = work_dir + '/confs'
preproc_dir = work_dir + '/glm_preproc'
fmri_glm_dir = work_dir + '/glm_data'
glm_sav_dir = work_dir + '/glm_out'
schaef_dir = work_dir + 'schaefer_atlas'
isc_dir = work_dir + '/isc_dat'


# In[ ]:





# In[108]:


# Functions 


# In[109]:


def resample_atlas(atlas_filename, mri_glm_dir):
    """
    purpose: resample yeo to MEI data
    input: 
    - atlas filename
    - location of preprocessed data for resampling
    output: 
    - atlas_img: the 3d brain image in numpy 2d
    - atlas_nii: the nifti image
    """
    # Load  sample data for resampling
##### MASKING #### 

    # Set the size (in terms of X, Y, Z) of the volume we want to create
    dimensions = np.asarray([78, 93, 65])

    # Generate an anatomical image with the size above of brain voxels in gray matter
    # This outputs variables for two versions of the image, binary (mask) and probabilistic (template)
    mask, template = sim.mask_brain(dimensions, mask_self=False)
    s_dat = np.load(f'{fmri_glm_dir}/sub-004_bpress_fmri_data.npy', allow_pickle=True).item() # load template data to get affine
    epi = s_dat['external']['sherlock']['f'][0]
    resamp_run  = nib.Nifti1Image(mask, epi.affine) ##create nifti image
    # Load parcellation
    d = nib.load(atlas_filename)
    atlas_nii = resample_to_img(d, resamp_run, interpolation='nearest')
    # Get parcellation fdata
    atlas_img = atlas_nii.get_fdata()
    # paracellations scheme
    print(f'count parc:{len(np.unique(atlas_nii.get_fdata()))}')
    print("shape of atlas nii object", atlas_img.shape)
    return atlas_nii, atlas_img




##### MASKING #### 

# Set the size (in terms of X, Y, Z) of the volume we want to create
dimensions = np.asarray([78, 93, 65])

# Generate an anatomical image with the size above of brain voxels in gray matter
# This outputs variables for two versions of the image, binary (mask) and probabilistic (template)
mask, template = sim.mask_brain(dimensions, mask_self=False)
s_dat = np.load(f'{fmri_glm_dir}/sub-004_bpress_fmri_data.npy', allow_pickle=True).item() # load template data to get affine
epi = s_dat['external']['sherlock']['f'][0]
mask_3d = nib.Nifti1Image(mask, epi.affine) ##create nifti image

##### Variables for ISC ### 

num_parc = 200 ## CHANGE ME
num_net= 17 ## CHANGE ME


###### ROI Loading ### 
# LOAD ATLAS #
## fetch dataset
dataset = datasets.fetch_atlas_schaefer_2018(n_rois=num_parc, yeo_networks = num_net)

# get nii dataset location
atlas_filename = dataset.maps
## get *ROI* atlas labels
labels = dataset.labels

# resample loaded atlas 
atlas_nii, atlas_img = resample_atlas(atlas_filename, fmri_prep)

# Load in network labels for each parcell, parcel UNspecific network labels, and the middle parcel within each network
networks, network_labels, network_idxs = get_network_labels(num_parc, num_net)





# Example placeholders for your actual data sources
conditions = ["ext", "int"]  # List of conditions
movies = ['office', 'brushing', 'oragami', 'shrek', 'cake', 'sherlock']
runs = [1, 2, 3, 4]                          # List of runs

# Initialize an empty list to store rows of the DataFrame
data_rows = []

# Iterating through all combinations
for condition in conditions:
    all_dat = np.load(f'{isc_dir}/n39_{condition}_isc.npz')       
    for movie in movies:
        print(f'\n****loading movie {movie}...***\n')
        data = all_dat[movie]
        for run in runs:
            
            # Z-score time series for each voxel
            data_ = zscore(data[...,run - 1,:], axis=0)
            
            # Leave-one-out approach
            print(f'isc...')
            iscs = isc(data_, pairwise=False, tolerate_nans=.8)
            subjects = iscs.shape[0]
            
            for subject in range(subjects):
                print(f'sub {subject+1} of {iscs.shape[0]}')
                # Extract data (replace this with your actual logic)
                im_3d = masking.unmask(iscs[subject,:], mask_3d)
                im_1d = im_3d.get_fdata()
                for roi in np.arange(200):
                    print('roi:', roi)
                    temp = np.nanmean(im_1d[atlas_img == roi+1])
                    # Append the row to the list
                    data_rows.append({
                        "Subject": subject,
                        "cond": condition,
                        "mov": movie,
                        "Roi": roi,
                        "run": run,
                        "isc_val": temp  # Add the extracted data to a column
                    })

# Create a DataFrame from the list of rows
df = pd.DataFrame(data_rows)



# SAVE #


high_sal = ['shrek', 'sherlock', 'office']
df['salience'] = df['mov'].apply(lambda mov: 'high' if mov in high_sal else 'low')
df.to_csv(os.path.join(isc_dir,'isc_ALL_roi_values.csv'))

