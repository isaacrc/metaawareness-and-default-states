#!/usr/bin/env python
# coding: utf-8


####### DESCRIPTION #######
# This script takes the preprocessed images output from preproc1.py and makes them cool -- i.e. reshapes them, and organizes them
# into condition type, movie

### Imports 


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


from nilearn.image import concat_imgs, resample_img, mean_img,index_img
from nilearn import image
from nilearn import masking
from nilearn.plotting import view_img
from nilearn.image import resample_to_img

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


# Seed 


random.seed(10)


# ## custom helper functions 
from utils_anal import load_epi_data

# ## directories 
top_dir = '../../../' ## CHANGE ME ## 
data_dir = top_dir + "/data"
work_dir = data_dir + '/work'
mask_dir = work_dir + '/masks'
behav_dir = top_dir + '/data/behavioral'
rois_dir = data_dir + "/rois"
fmri_prep = data_dir + '/bids/derivatives/fmriprep'
conf_dir = work_dir + '/confs'
preproc_dir = work_dir + '/glm_preproc'

### sub_list ###



## ALL SUBLIST
sub_list = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005','sub-006','sub-007','sub-008','sub-009','sub-010', 'sub-012',
    'sub-013','sub-014', 'sub-015', 'sub-016','sub-017','sub-018','sub-019','sub-020','sub-021',
    'sub-022','sub-023','sub-024','sub-025','sub-026','sub-027','sub-028', 'sub-029', 'sub-030','sub-031','sub-032',
    'sub-033','sub-034','sub-035','sub-036','sub-037','sub-038','sub-039','sub-040', 'sub-041'
]



## function to be used below ## 


# ## custom helper functions 
def reshape_dictionary(data, num_repeats = 4):
    """
    purpose: create a 4d array sorted into x runs per subject
    input: dictionary of stacked subject data 
    output: 4d array sorted into runs
    """
    output_dict = {}
    for mov in data:
        print(f'{mov} size is {data[mov].shape}')
        dim_3 = num_repeats
        dim_4 = int(data[mov].shape[2] / num_repeats)
        
        if key not in output_dict:
            output_dict[mov] = data[mov].reshape(*data[mov].shape[:-1], dim_3, dim_4, order = 'F')
        
        #print(np.array_equal(data[mov][:, :, 1], output_dict[mov][:, :, 0, 1]))
        #print(np.array_equal(data[mov][:, :, 17], output_dict[mov][:, :, 1, 0]))
        print(np.array_equal(data[mov][:, :, 4], output_dict[mov][:, :, 0, 1]))
        print(f'reshaped {output_dict[mov].shape}')
    print()
    return output_dict



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


##### Variables for preproc ### 
prefix = 'n39'

## load the runs to be included for each subject ## 
sub_run_inc = np.load(behav_dir + '/sub_run_inc.npy', allow_pickle = True).item()

## how many TRs on the end? ## 
tr_buffer = 4 

## how many TRs to trim from front? 
tr_trim = 4 

## how man EPI runs
epi_runs = 6
start_rep = 1 
end_rep = 4


### static vars ### 
run_list = [1, 2, 3, 4]
cond_list = ["internal", "external"]
mov_list = ['office', 'brushing', 'oragami', 'shrek', 'cake', 'sherlock']




######## BEGIN ########
# set empty dic
external = {}
internal = {}

for sub in sub_list:
    ### fMRI load ###
    sub_dic_fmri = np.load(f'{preproc_dir}/{sub}_raw_4D.npy', allow_pickle=True).item()
    print(f'start {sub}')
    ## BEHAVIORAL ##
    sub_dic_behav = np.load(os.path.join(behav_dir, f'{sub}_behav.npy'), allow_pickle=True).item()

    # Initialize an empty dictionary to store the stacked arrays for the current subject
    stacked_arrays = {}
    
    # Create subject number 
    sub_num = int(sub[-3:])

    for epi_index in range(0, epi_runs):
        # Add one to the index to create 1-6 runs
        epi_run = epi_index + 1
        
        # check if run is to be included 
        if not sub_run_inc[sub][epi_run]: continue

        # Get the movie name
        mov_name = sub_dic_behav['mov_order'][epi_index]

        # Create an empty array for the movie runs
        mov_runs = np.zeros((mov_len_dic[mov_name] + tr_buffer - tr_trim, voxel_num, 0))
        mov_runs = []

        print(f'movie: {mov_name}')

        # Get the fMRI run for the current epi_index
        fmri_run = sub_dic_fmri[epi_run]
        
        # Loop over runs
        for run in range(start_rep, end_rep + 1):
            # Is this an internal or external run?
            key = 'External' if (sub_num % 2 == 1 and epi_index < 3) or (sub_num % 2 == 0 and epi_index >= 3) else 'Internal'
            

            # Begin slicing #
            start_tr = sub_dic_behav[key][mov_name][f'run-{run:d}']['start_tr']
            end_tr = sub_dic_behav[key][mov_name][f'run-{run:d}']['end_tr']
            #run_slice = fmri_run[(start_tr + tr_trim):end_tr, :]
            run_slice = index_img(fmri_run, slice((start_tr + tr_trim), end_tr))
            print(f'rep {run} in {mov_name} is: {run_slice.shape}')
            
            # BEHAV checks #
            #assert mov_runs.shape[0] == mov_len_dic[mov_name] + tr_buffer - tr_trim, f'behavioral ERROR: movie: {mov_name}, epi {epi_run}, run: {run}'

            # fMRI data check #
            #assert run_slice.shape[3] == mov_runs.shape[0] + tr_buffer - tr_trim, f'fMRI error! {run_slice.shape[3]} verse {mov_runs.shape[0]}'
        
            # Stack the run slice with the mov_runs array
            mov_runs.append(run_slice)
            print(f'stacked! length is: {len(mov_runs)}')

        
        # set outer loop #
        if key == 'External':
            target_dict = external
        else:
            target_dict = internal

        if mov_name not in target_dict:
            target_dict[mov_name] = mov_runs
        else:
            #mov_runs = np.expand_dims(mov_runs, 3)
            target_dict[mov_name] = np.dstack((target_dict[mov_name], mov_runs))
    print(f'\n subject {sub} finished \n')

            
print('done')
                 
## Reshape Dictionary
ext_isc = reshape_dictionary(external)
int_isc = reshape_dictionary(internal)

## save if you'd like 
np.savez_compressed(f'{preproc_dir}/{prefix}_ext_isc.npz', **ext_isc)
np.savez_compressed(f'{preproc_dir}/{prefix}_int_isc.npz', **int_isc)




