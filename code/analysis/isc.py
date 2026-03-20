#!/usr/bin/env python
# coding: utf-8


####### DESCRIPTION #######
# This script takes the preprocessed images output from slurm_create-data_preproc.ipynb and makes them cool -- i.e. reshapes and performs ISC

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

from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import make_first_level_design_matrix
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

# Brainiak # 
from brainiak import image, io 
from brainiak.isc import (isc, isfc, bootstrap_isc, permutation_isc,
                          timeshift_isc, phaseshift_isc,
                          compute_summary_statistic)
from brainiak.io import load_boolean_mask, load_images
from statsmodels.stats.multitest import multipletests
from nilearn.plotting import plot_stat_map

# Seed 


random.seed(10)


# ## custom helper functions 
from utils_anal import load_epi_data, resample_atlas, get_network_labels

# ## directories 
top_dir = ... ## CHANGE ME ## 
data_dir = top_dir + "/data"
work_dir = data_dir + '/work'
mask_dir = work_dir + '/masks'
behav_dir = top_dir + '/data/behavioral'
rois_dir = data_dir + "/rois"
fmri_prep = data_dir + '/bids/derivatives/fmriprep'
conf_dir = work_dir + '/confs'
preproc_dir = work_dir + '/preproc'
isc_dir = work_dir + '/isc_dat'

### sub_list ###



## ALL SUBLIST
sub_list = [
    'sub-002', 'sub-003', 'sub-004', 'sub-005','sub-006','sub-007','sub-008','sub-009','sub-010', 'sub-012',
    'sub-013','sub-014', 'sub-015', 'sub-016','sub-017','sub-018','sub-019','sub-020','sub-021',
    'sub-022','sub-023','sub-024','sub-025','sub-026','sub-027','sub-028', 'sub-029', 'sub-030','sub-031','sub-032',
    'sub-033','sub-034','sub-035','sub-036','sub-037','sub-038','sub-039','sub-040', 'sub-041'
]




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
prefix = 'n39' ### CHANGE ME 

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


##### Variables for ISC ### 
n_boot = 1000 # how many boostraps?
p_thresh = .05

### static vars ### 
run_list = [1, 2, 3, 4]
cond_list = ["internal", "external"]
mov_list = ['office', 'brushing', 'oragami', 'shrek', 'cake', 'sherlock']
thresh_vis_dic = {cond: {mov: {run: {} for run in run_list} for mov in mov_list} for cond in cond_list}
nothresh_vis_dic = {cond: {mov: {run: {} for run in run_list} for mov in mov_list} for cond in cond_list}


## mask info ##
ref_nii = nib.load(mask_dir + "/whole_b_bnk.nii.gz")
mask_img = load_boolean_mask(mask_dir + "/whole_b_bnk.nii.gz")
mask_coords = np.where(mask_img)


## LOADDDD ##
## be sure to run preprocessing scripts first 
ext_isc = np.load(f'{isc_dir}/{prefix}_ext_isc.npz')
int_isc = np.load(f'{isc_dir}/{prefix}_int_isc.npz')


################### ISC ANAL ################## #

for cond in cond_list:
    print(f'START {cond}')
    if cond == "external":
        targ_dic = ext_isc
    else:
        targ_dic = int_isc
    for run in run_list:
        for mov in mov_list:
    
            # select movie and run from loaded data ** account for indexing
            data = targ_dic[mov][...,run - 1,:]
            print(f'{mov} run {run} shape is: {data.shape}')

            # Z-score time series for each voxel
            data = zscore(data, axis=0)

            # Leave-one-out approach
            iscs = isc(data, pairwise=False, tolerate_nans=.8)

            # Check shape of output ISC values
            print(f"ISC values shape = {iscs.shape} \ni.e., {iscs.shape[0]} "
                  f"left-out subjects and {iscs.shape[1]} voxel(s)")

            # Compute mean ISC (with Fisher transformation)
            mean_iscs = compute_summary_statistic(iscs, summary_statistic='mean', axis=0)

            print(f"ISC values shape = {mean_iscs.shape} \ni.e., {mean_iscs.shape[0]} "
                  f"mean value across left-out subjects and {iscs.shape[1]} voxel(s)"
                  f"\nMinimum mean ISC across voxels = {np.nanmin(mean_iscs):.3f}; "
                  f"maximum mean ISC across voxels = {np.nanmax(mean_iscs):.3f}")


            # Compute median ISC
            median_iscs = compute_summary_statistic(iscs, summary_statistic='median',
                                                    axis=0)

            print(f"ISC values shape = {median_iscs.shape} \ni.e., {median_iscs.shape[0]} "
                  f"median value across left-out subjects and {iscs.shape[1]} voxel(s)"
                  f"\nMinimum median ISC across voxels = {np.nanmin(median_iscs):.3f}; "
                  f"maximum median ISC across voxels = {np.nanmax(median_iscs):.3f}")

            # Run bootstrap hypothesis test on ISCs
            observed, ci, p, distribution = bootstrap_isc(iscs, pairwise=False,
                                                          ci_percentile=95,
                                                          summary_statistic='median',
                                                          n_bootstraps=n_boot)


            # Get number of NaN voxels
            n_nans = np.sum(np.isnan(observed))
            print(f"{n_nans} voxels out of {observed.shape[0]} are NaNs "
                  f"({n_nans / observed.shape[0] * 100:.2f}%)")

            # Get voxels without NaNs
            nonnan_mask = ~np.isnan(observed)
            nonnan_coords = np.where(nonnan_mask)

            # Mask both the ISC and p-value map to exclude NaNs
            nonnan_isc = observed[nonnan_mask]
            nonnan_p = p[nonnan_mask]

            # Get FDR-controlled q-values
            nonnan_q = multipletests(nonnan_p, method='fdr_by')[1]
            print(f"{np.sum(nonnan_q < p_thresh)} significant voxels "
                  f"controlling FDR at {p_thresh}")
            
            # non thresholded iscs
            nothresh_nonnan_isc = nonnan_isc.copy()
            # Threshold ISCs according FDR-controlled threshold
            nonnan_isc[nonnan_q >= p_thresh] = np.nan

            # Reinsert thresholded ISCs back into whole brain image
            isc_thresh = np.full(observed.shape, np.nan)
            isc_thresh[nonnan_coords] = nonnan_isc
            
            # Reinsert NON-thresholded ISCs back into whole brain image
            isc_nt = np.full(observed.shape, np.nan)
            isc_nt[nonnan_coords] = nothresh_nonnan_isc

            # Create empty 3D image and populate
            # with thresholded ISC values
            isc_img = np.full(ref_nii.shape, np.nan)
            isc_img[mask_coords] = isc_thresh
            
            # Create empty 3D image and populate
            # with NON-thresholded ISC values
            nt_isc_img = np.full(ref_nii.shape, np.nan)
            nt_isc_img[mask_coords] = isc_nt

            # Convert to NIfTI image
            isc_nii = nib.Nifti1Image(isc_img, ref_nii.affine, ref_nii.header)
            isc_nii_nt = nib.Nifti1Image(nt_isc_img, ref_nii.affine, ref_nii.header)
            
            # Save into dictionary
            thresh_vis_dic[cond][mov][run] = isc_nii
            nothresh_vis_dic[cond][mov][run] = isc_nii_nt
            print(f'\nfinish {cond}-{mov}-{run}\n')
            
            
            
            # DO rois # -- this is now completed in the visual script



                    
## save visualization for sig voxels
np.save(f'{isc_dir}/{prefix}_isc_sigvox_nii_dic.npy', thresh_vis_dic)
## save visualization for no threshold
np.save(f'{isc_dir}/{prefix}_isc_noThresh_nii_dic.npy', nothresh_vis_dic)
## save correlation dictionary
#np.save(f'{isc_dir}/{out_name}', cor_dic) -- see visual script 


