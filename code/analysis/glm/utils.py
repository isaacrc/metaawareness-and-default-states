import pandas as pd
import os
import numpy as np


data_dir = '../../../data/bids' # where the fMRI data is downloaded
rois_dir = ... # not specified 
behav_p = '../behavioral/stim_files/'
load_bpress = "../../../data/work/results/bpress_GLM/behav"
load_fmri = '../../../data/work/results/bpress_GLM/' #location of preprocessed data output from preproc scripts
out_dir = "../../../data/work/results/one_window/" # where to save the directory
confounds_dir = '../behavioral/confound_info/'
mask_dir = '../../../data/work/rois/'


def load_confounds(cond_list, sub_list,behav_p,confounds):
    """
    args: 
        cond_list: list of conditions (cond_list=np.array(['SM','SC']))
        sub_list: subjects to extract confounds for
        behav_p: path to the behavioral data
        confounds: path to the confound data
    returns:
        nested dictionary in the form of: conf_sub[sub][cond][img_ind]
        where img_index is the first or second run
    """
    # Confound files

    conf_sub = {}
    for sub in sub_list:
        conf_cond = {}
        for cond in cond_list:
            confs = []
            lab_indic = fnd_indices(sub, behav_p)
            confs.append(np.asarray(pd.read_csv(os.path.join(confounds + sub + "/func/", 
                                                             '%s_ses-01_task-Attn_run-%s_desc-model_timeseries.csv') % (sub, lab_indic[cond][0])))[4:,:])
            confs.append(np.asarray(pd.read_csv(os.path.join(confounds + sub + "/func/",
                                                             '%s_ses-01_task-Attn_run-%s_desc-model_timeseries.csv') % (sub, lab_indic[cond][1])))[4:,:])
            conf_cond[cond] = confs
        conf_sub[sub] = conf_cond
    return conf_sub



## Expand / Label TRs
"""
0 = SM
1 = SC
2 = OM
3 = OC
4 = Re
requires list of labels ouputed by psychopy (column 1 - MM_self_title.started, etc.)
returns label list (order is preserved) and TR labels
"""

def label_lists(label, num_tr):
    b = [[]]
    a = []
    for i in label:
        # substring label in psychopy output
        # if the first three characters == M_s, etc, then add correct indext to string
        if i[1:4] == "M_s":
            a.append("SM")
            b.append([0]*num_tr)
        elif i[1:4] == "C_s":
            a.append("SC")
            b.append([1]*num_tr)        
        elif i[1:4] == "M_o":
            a.append("OM")
            b.append([2]*num_tr)
        elif i[1:4] == "C_o":
            a.append("OC")
            b.append([3]*num_tr)     
        else:
            a.append("Re")
            b.append([4]*num_tr)     
    return a, b[1:]
 

def find_cond_index(sub_ses_labels):
    """
    For the array of ordered run names (i.e.'Re', 'SM',) find the two indexes per condition
    """ 
    lab_inx = []

    a = []
    b = []
    c = []
    d = []
    e = []

    for i in enumerate(sub_ses_labels):
        if i[1] == "SM":
            # append the index according to where it appeared in the array
            a.append(i[0])
        if i[1] == "SC":
            b.append(i[0])
        if i[1] == "OM":
            c.append(i[0])
        if i[1] == "OC":
            d.append(i[0])

    # Create a dictionary where each key contains the appropriate indexes
    lab_indic = {
        'SM' : a,
        'SC' : b,
        'OM' : c,
        'OC' : d,
        'RE' : [0,9]
    }
    return lab_indic 
    #np.vstack(lab_inx, ["SM", "SC", "OM", "OC"])

def org_bold_data(cond_data, labels, run_indexes, cond_a, cond_b): 
    """
    Organize the bold data into an array with four runs, two conditions # 
     - Get both indexes according to the appropriate label
     - Add cond a, then cond. Do this twice for both labels and bold data
     - lab_indic['SM'][i] evaluates to the first, then second index for each condition
     args:
         cond_data: list of 8 runs of bold data
         labels: label array for conditions
    returns:
        sequence of four runs of bold data, labels, for classification
    """
    bold_data = []
    label_data = []

    for i in range(0,2):
        print("condition: ", cond_a, "index: ", run_indexes[cond_a][i])
        print("condition: ", cond_b, "index: ", run_indexes[cond_b][i])
        bold_data.append(np.vstack((cond_data[run_indexes[cond_a][i]], cond_data[run_indexes[cond_b][i]])))
        label_data.append(np.hstack((labels[run_indexes[cond_a][i]], labels[run_indexes[cond_b][i]])))
    print("number of runs: ", len(bold_data), "bold shape: ", bold_data[0].shape, "label shape: ", label_data[0].shape)
    print("Expected shape: ", num_trs *2)
    print("Extracted: ", cond_a, cond_b)
    return bold_data, label_data

def load_epi_data(sub, ses, task,run, space):
  # Load MRI file
    if space == "MNI":
        epi_in = os.path.join(data_dir, sub, ses, 'func', "%s_%s_task-%s_run-%s_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz" % (sub, ses, task,run))
    elif space == "T1":
        epi_in = os.path.join(data_dir, sub, ses, 'func', "%s_%s_task-%s_run-%s_space-T1w_desc-preproc_bold.nii.gz" % (sub, ses, task,run))
    else:
        print("wrong load epi input. check this function")
    epi_data = nib.load(epi_in)
    print("Loading data from %s" % (epi_in))
    return epi_data

def load_roi_mask(rois_dir, ROI_name, space):
    if space == "MNI":
        maskdir = os.path.join(rois_dir)    
        print("expected shape: 78, 93,65")
    elif space == "T1":
        maskdir = os.path.join(rois_dir+ "/T1")
        print("expected shape: 56, 72,53")
    else:
        print("wrong mask input. check this function")
    # load the mask
    maskfile = os.path.join(maskdir, "%s.nii" % (ROI_name))
    mask = nib.load(maskfile)
    print("mask shape: ", mask.shape)
    print("Loaded %s mask" % (ROI_name))
    return mask

def intersect_mask(sub, num_runs,reg, ses="ses-01",task="Attn"):
    # This is based off of 'load_data' function in template
    # Loads all fMRI runs into a matrix #
    """
    reg = T1 or MNI registration?
    norm_type = by Space or by Time? 
    """
    yoz = []
    print("Begin intersecting, you sexy beast")
    for run in range(1, num_runs + 1):
        # Load epi data 
        epi = load_epi_data(sub,ses,task,run,reg)
        # Load ROI data
        roi_samp = compute_epi_mask(epi)
     #   print(roi_samp)
        #nifti_masker = NiftiMasker(mask_img=roi_samp)
        #maskedData = nifti_masker.fit_transform(epi)
        yoz.append(roi_samp)
    #print(concatenated_data)
    epi_data = nil.masking.intersect_masks(yoz)
    print("all done wit da intersextion (lol)")

    return epi_data

def load_fMRI3d(sub, num_runs,reg, norm_type, mask, ses="ses-01",task="Attn"):
    # This is based off of 'load_data' function in template
    # Loads all fMRI runs into a matrix #
    """
    reg = T1 or MNI registration?
    norm_type = by Space or by Time? - default is by space (rows)
    """
    concatenated_data = []
    
    for run in range(1, num_runs + 1):
        if sub == "sub-002":
            if run >=7:
                run = run+1
        if sub != "sub-010":
            # Load epi data 
            epi = load_epi_data(sub,ses,task,run,reg)
        else:
            # Load epi data 
            print("sub-10, watch out")
            bad_epi = load_epi_data(sub,ses,task,run,reg)
            good_epi = load_epi_data("sub-001",ses,task,run,reg)
            epi = resample_to_img(bad_epi , good_epi, interpolation='nearest')
        # delete first 9 TRs
        epi = index_img(epi,slice(4,210))
        
        # load confounds
        #run_conf = np.asarray(pd.read_csv(os.path.join(confounds + sub + "/func/", 
         #                                                  '%s_ses-01_task-Attn_run-%s_desc-model_timeseries.csv') % (sub, run)))
        print(epi.shape)
        # clean image
        # low_pass= .1, high_pass=1/128, .01 might be more normal...
        """
        clean_bold = image.clean_img(epi, standardize = False, #confounds = run_conf[5:], 
                                   t_r=1.5, low_pass= .1, high_pass = 1/128, mask_img = mask)
        
        # https://cdn.fs.pathlms.com/pOQYPTQ0TiupIUDUQYdV
        clean_bold = image.clean_img(epi, standardize = False, #confounds = run_conf[5:], 
                                   t_r=1.5, low_pass = .1, mask_img = mask)
        """
        
        #Smooth
        #clean_bold = image.smooth_img(clean_bold, fwhm=5)
        
        # F*k it mask off -- Load ROI data
        #roi_samp =load_roi_mask(ROI_name,reg)
        # Pull voxels from epi data # *** may need to change this 
        #nifti_masker = NiftiMasker(mask_img=roi_samp)
        #masked_data = nifti_masker.fit_transform(clean_bold)
        
        #append to cat_dat
        concatenated_data.append(epi)
    "FINISHED YAY BEAST"
    return concatenated_data

def fnd_indices(sub,behav_p):
    behav = pd.read_csv(os.path.join(behav_p, '%s_behav_cleaned.csv') % (sub))
    # Define the column in behav to be used for creating labels # 
    label = behav.iloc[:,1]
    # Create an array of labels [1] AND the order in which runs occured [0]#
    sub_ses_labels = label_lists(label, 200)
    ## Find run sequence, extraction condition indexes from behav data ## 
    return find_cond_index(sub_ses_labels[0])

def org_bdata(unsort_bdata, run_indexes, cond_a, cond_b): 
    """
    organize two runs for concatenation
    Two runs of cond_a, then two runs of cond_b
    
    """
    bold_data = []
    a = [unsort_bdata[run_indexes[cond_a][0]], unsort_bdata[run_indexes[cond_a][1]]]
    b = [unsort_bdata[run_indexes[cond_b][0]], unsort_bdata[run_indexes[cond_b][1]]]
    print("concatenated", cond_a, cond_b)
    return a + b

def create_event_list(sub, bpress, cond, run_dic, base_onset,comp_onset_list,stim_dur):
    """
    this function reads in a condition for each sub
    and returns the corresponding b4 + after events
    
    sub: subject number
    bpress: array of button press onset times
    cond: which condition do you want to create event dataframe fore
    """
    all_tims = []
    events = {}
    # Convert to array #
    bpress_arr = np.asarray(bpress[sub][cond])
    # Select runs to include according to run_dic, append
    all_tims = all_tims + list(bpress_arr[run_dic[sub][cond]])
    # how much to shift from button press onset


    # events will take  the form of a dictionary of Dataframes, one per run! 
    for run in range(len(all_tims)):
        cond_labs = []
        duration = []
        onsets = []
        window_list = []
        # Do base first, then iterate on windows # 
        cond_labs = cond_labs + ['win0']* len(all_tims[run])
        duration = duration + [stim_dur] * len(all_tims[run])
        # onsets 
        onsets_unshft_prs = list(all_tims[run].astype(float))
        onsets = onsets + [y + base_onset for y in onsets_unshft_prs]

        for idx, comp_onset in enumerate(comp_onset_list):
            # Create a list of 'press' for how many presses there were for each run
            # label evaluates to 'win1'
            cond_labs = cond_labs + ['win'+str(idx+1)]* len(all_tims[run])

            # How long does the button event event last? one tr, so 1.5
            duration = duration + [stim_dur] * len(all_tims[run])

            # these are the corresponding onset times
            onsets = onsets +  [y + comp_onset for y in onsets_unshft_prs]
            # how many windows? add to list
            window_list.append('win'+str(idx+1))
        # Define the events object
        events_ = pd.DataFrame(
            {'onset': onsets, 'trial_type': cond_labs, 'duration': duration})
        # remove the rest condition and insert into the dictionary
        events[run] = events_
    return events, window_list



def create_event_list_rest(sub, bpress, cond, cond_alt, run_dic, base_onset,comp_onset_list,stim_dur):
    """
    this function reads in a condition for each sub
    and returns the corresponding b4 + after events
    
    sub: subject number
    bpress: array of button press onset times
    cond: which condition do you want to create event dataframe fore
    """
    all_tims = []
    events = {}
    # Convert alternate bpress to array for ALTERNATE CONDITION #
    bpress_arr = np.asarray(bpress[sub][cond_alt])
    # Select runs to include according to run_dic, append
    # * include correct runs of rest - those that were'nt excluded due to hm  
    all_tims = all_tims + list(bpress_arr[run_dic[sub]['RE']])
    # how much to shift from button press onset
    
    ## create fake bpress for missing data 
    ## if only one run for SM, then we still need button press data for two runs of RE ## 
    ## only add a second run if both runs of RE are included [second condition]
    if len(all_tims) <2 and len(run_dic[sub]['RE']) >1:
        all_tims = list(all_tims[0], all_tims[0])
        

    # events will take  the form of a dictionary of Dataframes, one per run! 
    for run in range(len(all_tims)):
        cond_labs = []
        duration = []
        onsets = []
        window_list = []
        # Do base first, then iterate on windows # 
        cond_labs = cond_labs + ['win0']* len(all_tims[run])
        duration = duration + [stim_dur] * len(all_tims[run])
        # onsets 
        onsets_unshft_prs = list(all_tims[run].astype(float))
        onsets = onsets + [y + base_onset for y in onsets_unshft_prs]

        for idx, comp_onset in enumerate(comp_onset_list):
            # Create a list of 'press' for how many presses there were for each run
            # label evaluates to 'win1'
            cond_labs = cond_labs + ['win'+str(idx+1)]* len(all_tims[run])

            # How long does the button event event last? one tr, so 1.5
            duration = duration + [stim_dur] * len(all_tims[run])

            # these are the corresponding onset times
            onsets = onsets +  [y + comp_onset for y in onsets_unshft_prs]
            # how many windows? add to list
            window_list.append('win'+str(idx+1))
        # Define the events object
        events_ = pd.DataFrame(
            {'onset': onsets, 'trial_type': cond_labs, 'duration': duration})
        # remove the rest condition and insert into the dictionary
        events[run] = events_
    return events, window_list



