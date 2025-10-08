# Scripts
* preproc_1: cleans raw .nii files output from fMRI prep
* preprop_2: organizes output from preproc_1 into an experimentally relevant dictionary (repetition, condition)
* preproc_by_sub.py: preprocesses subject data preserving subject level detail. Useful for ISC stats in unthresh.ipynb script
* extract_confounds.ipynb: Extract confounds from fmriprep
* isc-new.py: iterates through preprocessed data from slurm_create-data_preproc.ipynb and creates a dictionary of files organized by run, movie, condition. Then runs ISC
* isc-vis: plot the ISC results 
* utils_anal.py: utils used for analyses
* prepare_behav_data_6-8-23: cleans raw psychopy data 
* bpress_analysis-6-22-23: Plots button press data and runs stats
* wsfc_3.ipynb: Runs within-subject's functional connectivity analysis
* unthresh_isc.ipynb: calculate unthresholded ISC, statistics for unthresh ISC, get repetition 4 minus repetition 1 differences
* thresh_isc.ipynb: plot for thresholded isc
