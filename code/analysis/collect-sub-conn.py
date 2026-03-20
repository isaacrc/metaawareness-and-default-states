#!/usr/bin/env python
# coding: utf-8

# # Run ISPC
"""
 This script runs ISPC on all runs, movies, and ROIs provided. Result is a HYUGE dictionary:
 roi_bpress_ispc[region][cond][run][key][movie]
"""

# ## py conversion

# In[18]:


#!jupyter nbconvert --to python ISPC.ipynb


# ## Imports 

# In[2]:


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


# In[3]:


random.seed(10)


# ## custom helper functions 

# In[4]:


from utils_anal import resample_atlas, get_network_labels


# ## directories 


top_dir = '/jukebox/graziano/coolCatIsaac/mei'
data_dir = top_dir + "/data"
work_dir = data_dir + '/work'
behav_dir = top_dir + '/data/behavioral'
rois_dir = data_dir + "/rois"
fmri_prep = data_dir + '/bids/derivatives/fmriprep'
conf_dir = work_dir + '/confs'
preproc_dir = work_dir + '/preproc'
bpress_dir = work_dir + '/button_press_counts'
wsfc_dir = work_dir + '/wsfc'


##### Variables for the atlas ### 

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
dimensions = np.asarray(atlas_nii.shape)

# Load in network labels for each parcell, parcel UNspecific network labels, and the middle parcel within each network
networks, network_labels, network_idxs = get_network_labels(num_parc, num_net)

###############
# FUNCTIONS ###

def convert_2d_to_schaef(_2d_mat):
    """
    purpose: get brain image and convert to parcels
    input: takes in a 4d mat from get_fdata()
    ouptut: TRs x Parcels
    """
    ## brain mask  ##
    mask, template = sim.mask_brain(dimensions, mask_self=False)
    mask_nifti = nib.Nifti1Image(mask.astype(np.uint8), affine=atlas_nii.affine)

    ref_nii = mask_nifti ## reference image resampled to our space -- aquired above

    ## Take the 2d TR x voxels and convert it to a 4D fMRI image 
    new_nii = unmask(_2d_mat, ref_nii)
    # convert to 4d numpy
    func_data = new_nii.get_fdata()
    ## change 4d back to 2
    func_means = [np.mean(func_data[atlas_img == parcel, :], axis=0)
                  for parcel in np.unique(atlas_img)[1:]]
    func_parcels = np.column_stack(func_means)
    print(f'converted: {func_parcels.shape}')
    return func_parcels

def wsfc_mat_fish_1sub(img_2d, num_parc):
    """
    purpose: compute wsfc matrices
    input: TR x vox x subs [e.g. (98, 112179]
    ouput: parc x parc x sub [e.g. (200, 200]
    """

    print('correlating...')
    img_2d_schaef = convert_2d_to_schaef(img_2d)
    print(f'transformed: {img_2d_schaef.shape}')
    ## get correlation matrix
    cor_mat = np.corrcoef(img_2d_schaef.T)
    print(cor_mat.shape) 
    reshape_mat_fish = np.arctanh(cor_mat) #### FISHER transform
    print('fisher transform, now: ', reshape_mat_fish.shape)
    return reshape_mat_fish

##########


## ALL SUBLIST -- exclude sub-014 and sub-020 cuz no button presses...
# also sub-015... no idea why its not stacking.. can try to debug..?
sub_list = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005','sub-006','sub-007','sub-008','sub-009','sub-010', 'sub-012',
    'sub-013', 'sub-016','sub-017','sub-018','sub-019','sub-020','sub-021',
    'sub-022','sub-023','sub-024','sub-025','sub-026','sub-027','sub-028', 'sub-029', 'sub-030','sub-031','sub-032',
    'sub-033','sub-034','sub-035','sub-036','sub-038','sub-039','sub-040', 'sub-041'
]
#sub_list= ['sub-015']

###### LOADING VARS #######


## TR length of each movie ## 
mov_len_dic = {
'oragami' :  82,
'shrek' : 90,
'sherlock' : 98,
'brushing' : 88,
'cake' : 99,
'office' : 102    
}

voxel_num = 112179


#######################
### VARIABLES ## 
#######################

# output name
out_name = 'ext_int_sub_cor_mat.npy' # CHANGE ME 

# which movie repetitions
start_rep = 1
end_rep = 4 


## #### Static vars ##### ##

## how many TRs of buffer on the end? ## 
tr_buffer = 4 

# cuttoff the countdown -**effectivly shifts the timeseries over by 4
trim_start = 4 # cuts off first 4 TRs 

## number of runs to iterate over
epi_runs = 6

## load the runs to be included for each subject ## 
sub_run_inc = np.load(behav_dir + '/sub_run_inc.npy', allow_pickle = True).item()


## create dictionaries fore each subject
s_dat = {}



for sub in sub_list:
    ### fMRI load ###
    sub_dic_fmri = np.load(f'{preproc_dir}/{sub}_fwhm6_conf.npy', allow_pickle=True).item()
    print(f'start {sub}')
    ## BEHAVIORAL ##
    sub_dic_behav = np.load(os.path.join(behav_dir, f'{sub}_behav.npy'), allow_pickle=True).item()

    # Create subject number 
    sub_num = int(sub[-3:])

    cond_dic = {} # Set cond_dic
    internal = {} # Set cond_dic
    external = {} # Set cond_dic
    for epi_index in range(0, epi_runs):
        # Add one to the index to create 1-6 runs
        epi_run = epi_index + 1

        # check if run is to be included 
        if not sub_run_inc[sub][epi_run]: continue

        # Get the movie name
        mov_name = sub_dic_behav['mov_order'][epi_index]

        # Create an empty array for the movie runs, append four TRs to account for the 4 trailing TRs, subtract
        # the quantity of TRs that we are trimming from the front 
        #mov_runs = np.zeros((range_len, voxel_num, 0))

        print(f'movie: {mov_name}')

        # Get the fMRI run for the current epi_index
        fmri_run = sub_dic_fmri[epi_run]

        # Loop over repetitions
        # Is this an internal or external run?
        key = 'External' if (sub_num % 2 == 1 and epi_index < 3) or (sub_num % 2 == 0 and epi_index >= 3) else 'Internal'

        ## create dictionary to store dictionary
        repetition = {}

        ## Loop through Data ## 
        for run in range(start_rep, end_rep + 1):

            ## get behavioral data
            bpress_arr = sub_dic_behav[key][mov_name][f'run-{str(run)}']['bpress']

            ## continue if no button presses
            if bpress_arr == -1:
                print(f'NO button presses {sub} {run}\n')
                continue
                



            # Begin slicing fMRI data #
            start_tr = sub_dic_behav[key][mov_name][f'run-{run:d}']['start_tr']
            end_tr = sub_dic_behav[key][mov_name][f'run-{run:d}']['end_tr']
            run_slice = fmri_run[(start_tr + trim_start):end_tr, :]
            print(f'run{run} bpress count is: {len(bpress_arr)}')
            print(f'start tr {start_tr}, end TR {end_tr}, length of fMRI run {fmri_run.shape}')
            assert fmri_run.shape[0] >= end_tr, 'end TR is greater than fMRI TRs available'
            print(f'RUN SHAPE: {run_slice.shape}')
            
            cor_mat = wsfc_mat_fish_1sub(run_slice, num_parc)
            repetition[run] = cor_mat # repetition = {1 : img, 2: img}

        if not repetition:
            continue
            
        # set outer loop #
        print(f'reptitions dic: {list(repetition.keys())}')
        if key == 'External':
            ## last run minus the first run
            external[mov_name] = repetition[list(repetition.keys())[-1]] - repetition[list(repetition.keys())[0]] ## SUBTRACT
        else:
            internal[mov_name] = repetition[list(repetition.keys())[-1]] - repetition[list(repetition.keys())[0]]

    print(f'conditions dic: {list(external.keys())}')

    ## save into a repetition dictionary ## 
    cond_dic['external'] = external # external = {'shrek' : repetition, 'sherlock' : repetition}
    cond_dic['internal'] = internal
    
    averaged_dic = {} # Initialize an empty dictionary to hold the stacked and averaged matrices across MOVIE
    for cond_key in ['external', 'internal']:
        matrices = []  # List to hold matrices for stacking

        for key in cond_dic[cond_key]:
            matrices.append(cond_dic[cond_key][key])  # Append each matrix
        
        print(cond_key)
        print(np.array(matrices).shape)
        if not matrices:
            continue
        # Stack the matrices along a new dimension (axis=0)
        stacked_matrices = np.stack(np.array(matrices), axis=0)

        # Average along the stacked dimension (axis=0)
        averaged_matrix = np.mean(stacked_matrices, axis=0)

        # Store the averaged matrix in the new dictionary
        averaged_dic[cond_key] = averaged_matrix
    
    # The averaged_dic will now hold the averaged matrices for 'external' and 'internal'
    #print("Averaged external matrix:\n", averaged_dic['external'].shape)
    #print("Averaged internal matrix:\n", averaged_dic['internal'].shape)
        
    ## save subject ## 
    s_dat[sub] = averaged_dic # sub01 = {'internal' : averaged_cor_mat_across_movies, 'external' : xxx}
    print(f'finish {sub}')

    ### save ##
    print('saving...')
    np.save(f'{wsfc_dir}/{out_name}', s_dat)
print('FINNNISSHHHHEDD AHAHAHA ')


            

    
        

