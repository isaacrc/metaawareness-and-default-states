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



# In[3]:


random.seed(10)


# ## custom helper functions 

# In[4]:


from utils_anal import load_epi_data, load_conf_data


# ## directories 


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

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(preproc_dir):
    os.makedirs(preproc_dir)
    print(f"Directory created: {preproc_dir}")


## ALL SUBLIST
'''
NO DATA FOR SUB-015??? 
'''

sub_list = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005','sub-006','sub-007','sub-008','sub-009','sub-010', 'sub-012',
    'sub-013','sub-014', 'sub-015', 'sub-016','sub-017','sub-018','sub-019','sub-020','sub-021',
    'sub-022','sub-023','sub-024','sub-025','sub-026','sub-027','sub-028', 'sub-029', 'sub-030','sub-031','sub-032',
    'sub-033','sub-034','sub-035','sub-036','sub-037','sub-038','sub-039','sub-040', 'sub-041'
]

sub_list = ['sub-032']

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

"""
**************************************
#********* IMPORTANT NOTE ************#
**************************************
~~~ TO PUT IT SIMPLY: I SHIFT THE FMRI DATA BACK FOUR TRS TO ACCOUNT FOR THE HEMODYNAMIC LAG.~~~

I TRIMMED FOUR TRS FROM THE NEURAL DATA, BUT THE BUTTON PRESS IS STILL TIMELOCKED TO THE START OF THE 
VIDEO, THUS IT INCLUDES THE 7.5 SECONDS [5 TRS!]
SINCE I TRIM 4, AND KEEP THE BPRESS TIMESERIES THE SAME, I AM ESSENTIALLY TIMESHIFTING, ACCOUNTING FOR THE HEMODYNAMIC LAG
BY SHIFTING THE FMRI DATA BACK SIX SECONDS
"""
#######################
### VARIABLES ## 
#######################

# output name
out_name = 'glm_data_test' # CHANGE ME 

# which movie repetitions
start_rep = 1
end_rep = 4 


## #### Static vars ##### ##

## how many TRs of buffer on the end? ## 
tr_buffer = 4 

# cuttoff the countdown -**effectivly shifts the timeseries over by 4
trim_start = 0 # cuts off first 4 TRs. we want this to be 0 i think? Time 0 is the start of the 7.5 second countdown

## number of runs to iterate over
epi_runs = 6

## load the runs to be included for each subject ## 
sub_run_inc = np.load(behav_dir + '/sub_run_inc.npy', allow_pickle = True).item()


#######################
### LOADING ## 
#######################

## create dictionaries fore each subject
s_dat = {}



for sub in sub_list:
    ### fMRI load -- this is the fMRI with 6 EPI runs. Each run is one movie of four runs ###
    sub_dic_fmri = np.load(f'{preproc_dir}/{sub}_raw_4D.npy', allow_pickle=True).item()
    
    print(f'start {sub}')
    ## BEHAVIORAL ##
    sub_dic_behav = np.load(os.path.join(behav_dir, f'{sub}_behav.npy'), allow_pickle=True).item()

    # Create subject number 
    sub_num = int(sub[-3:])

    cond_dic = {} # Set cond_dic
    internal = {} # Set cond_dic
    external = {} # Set cond_dic
    for epi_index in range(0, epi_runs):
        print(f'\n***STARTING EPI RUN {epi_index}***')
        # Add one to the index to create 1-6 runs
        epi_run = epi_index + 1

        # check if run is to be included 
        if not sub_run_inc[sub][epi_run]: continue
            
        # load confounds 
        run_conf = load_conf_data(conf_dir, sub, epi_run)

        # Get the movie name
        mov_name = sub_dic_behav['mov_order'][epi_index]

        # Create an empty array for the movie runs, append four TRs to account for the 4 trailing TRs, subtract
        mov_runs = [] # movies
        b_runs = [] # bpress
        c_runs = []# confounds

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
            
            # TRIIM
            # trim_start should be TR 0 for button press s- exactly when the screen flipped 
            run_slice = index_img(fmri_run, slice((start_tr + trim_start), end_tr))
            print(f'run{run} bpress count is: {len(bpress_arr)}')
            print(f'start tr {start_tr}, end TR {end_tr}, length of fMRI run {fmri_run.shape[3]}')
            conf_temp = run_conf[(start_tr + trim_start): end_tr, :]
            assert fmri_run.shape[3] >= end_tr, 'end TR is greater than fMRI TRs available'
            
            # Add fmri run
            mov_runs.append(run_slice)
            
            # add bpress
            b_runs.append(bpress_arr)
            
            # add confounds
            c_runs.append(conf_temp)

            # calculate differences between button presses, and append the time stamp of the end of scan
            difs = np.diff(np.hstack((bpress_arr, (end_tr - start_tr) * 1.5)))
            print(f'button presses: {bpress_arr}')
                        
        assert len(mov_runs) == len(b_runs)
        
        
        ### Store button presses as 'b' and fMRI runs as 'f' and confounds as 'c' in a dictionary
        four_rep_dic = {'b' : b_runs,
                        'f' : mov_runs,
                        'c' : c_runs
                       }


        # set outer loop #
        #print(f'reptitions dic: {four_rep_dic.keys()}')
        if key == 'External':
            external[mov_name] = four_rep_dic
        else:
            internal[mov_name] = four_rep_dic

    #print(f'conditions dic: {external.keys()}')
    ## save into a repetition dictionary ## 
    cond_dic['external'] = external # external = {'shrek' : four_rep_dic, 'sherlock' : four_rep_dic}
    cond_dic['internal'] = internal
        
    ## save subject ## 
    s_dat[sub] = cond_dic # sub01 = {'internal' : internal, 'external' : external}
    
    ## saving ## 
    np.save(f'{fmri_glm_dir}/{sub}_bpress_fmri_data.npy', cond_dic)
    print(f'finish {sub}')
    
### save all of it##
#print('saving...')
#np.savez_compressed(f'{work_dir}/{out_name}.npz', data=s_dat)
#print('saving complete ')


            

    
        

