# import python modules to use for analysis
import numpy as np
import pandas as pd
import seaborn as sns
import os
import subprocess
import glob
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import ast
import warnings


# start defining some functions

# function for fetching the data from the internet

def check_data(project_id):
    # check that csv files have been downloaded
    path = "{}/osfstorage/data".format(project_id)
    ext = "*.csv"
    try:
        all_csv_files = [file
                    for path, subdir, files in os.walk(path)
                    for file in glob.glob(os.path.join(path, ext))]
        num_files = len(all_csv_files)
    except:
        num_files = 0
    
    return(num_files)

def fetch_data(project_id):
    num_files = check_data(project_id)
    if num_files < 10:
        str_to_run = "osf --project {} clone".format(project_id)

        try:
            subprocess.check_call("{0}".format(str_to_run), shell=True, stderr=subprocess.STDOUT, universal_newlines=True, encoding='utf8')
        except subprocess.CalledProcessError:
            print("fetch command failed: \n {}".format(str_to_run))

        # check again
        assert check_data(project_id) >= 10, "less than 10 datasets were downloaded from project {} \n".format(project_id)
    else:
        print("data already exists, not downloading")

# function for loading the data into memory

def load_data(data_dir, grab_course = "all", grab_term = "all", labels=['bear','elephant','person','car','dog','apple','chair','plane','bird','zebra'], exp_n = 199):
    if not isinstance(data_dir, list):
        data_dir = [data_dir]

    # define empty lists that will hold the number of subjects, rejected subjects and test subjects
    sub_count = [0]*len(data_dir)    # included datasets
    reject_count = [0]*len(data_dir) # complete datasets, but rejected due to performance
    test_count = [0]*len(data_dir)   # incomplete test datasets

    complete_subs = [] # subjects that produced complete datasets

    all_data = []

    for e, cur_dir in enumerate(data_dir):
        file_list = glob.glob(cur_dir + "/**/psyc4260*.csv") + glob.glob(cur_dir + "/**/nrsc2200*.csv")
        if grab_course not in ["all", "ALL", "All"]:
            file_list = [x for x in file_list if grab_course.upper() in x.upper() ]
            print("grabbing data from course {}".format(grab_course.upper()))
        else:
            print("grabbing data from all courses")
        if grab_term not in ["all", "ALL", "All"]:
            file_list = [x for x in file_list if grab_term.upper() in x.upper() ]
            print("grabbing data from term {}".format(grab_term.upper()))
        else:
            print("grabbing data from all terms")
        assert len(file_list) > 0, "No data found for course {} in term {}".format(grab_course.upper(), grab_term.upper())
        file_list.sort()
        sub_list = [] # list to hold the subjects in this experiment
        for file in file_list:
            # load the data
            try:
                sub_data = pd.read_csv(file)
                if "trial_type" not in sub_data:
                    sub_data = pd.read_csv(file, skiprows=1)
                # in the past, some trials were duplicated in the data file, the code below takes care of that
                sub_data = sub_data[sub_data['trial_index'].apply(lambda x: str(x).isdigit())]
                sub_data = sub_data.drop_duplicates()
            except:
                print("Failed to load file {0}".format(file.split(cur_dir)[1]))
            # get id
            try:
                survey_resp = sub_data[sub_data["trial_type"]=="survey-html-form"]["responses"].values[0]
                survey_resp = survey_resp.replace(':"}',':""}')
                sub_info = ast.literal_eval(survey_resp)
            except:
                sub_info = {}
            # see if id was stored
            if 'p_id' in sub_info.keys():  
                sub_id = sub_info["p_id"]
            else:
                sub_id = "nan"
                
            # do quality control on the data
            if sub_data.shape[1] < 10:
                print(e, sub_id, "incomplete file, shape:{0}x{1}".format(sub_data.shape[0],sub_data.shape[1]))
                test_count[e] = test_count[e] + 1 # record test subject to the test_count, by adding 1 at the relevant position
                continue
            
            # now skip if this subjects data is already in the set
            if sub_id in sub_list:
                print(e, sub_id, "duplicate participants, skipping the second file".format(num_trials))
                continue
            
            # now we can start working with the data
            images = sub_data["images"]
            left_choice = sub_data["left_choice"]
            right_choice = sub_data["right_choice"]
            rts = sub_data["rt"]
            response = sub_data["button_pressed"] # 0 = left; 1 = right;
            
            valid_loc = [ ~np.isnan(x) for x in images ]
            valid_images = [ int(x) for i, x in enumerate(images) if valid_loc[i] ] # image number
            valid_left = [ int(x) for i, x in enumerate(left_choice) if valid_loc[i] ] # which object appeared as left choice
            valid_right = [ int(x) for i, x in enumerate(right_choice) if valid_loc[i] ]
            valid_response = [ int(response[i+1]) for i, x in enumerate(response) if valid_loc[i] ] # which side did the subject choose
            choices = [ [l, r] for l, r in zip(valid_left, valid_right) ]
            valid_rt = [ float(rts[i+1]) for i, x in enumerate(rts) if valid_loc[i] ] # reaction time
            
            correct_obj_no = np.tile(range(10),[20,1])
            correct_obj_no = correct_obj_no.flatten("F") # corresponding correct object number per image
            correct_choice = [ correct_obj_no[x] for x in valid_images ]
            
            # final steps of quality control - do we actually have the expected number of trials?
            num_trials = sum(valid_loc) # total trials
            if num_trials < exp_n:
                print(e, sub_id, "incomplete file, on {0} trials".format(num_trials))
            
            # add participant to list of subjects for this experiment
            sub_list.append(sub_id)
            
            # start populating sub_info dict: 
            sub_info["experiment"] = e
            sub_info["ID"] = sub_id
            
            # lets fetch all the relevant trial variables for each trial
            sub_info["im_no"] = valid_images; # all image numbers
            sub_info["target"] = correct_choice; # what was the target object number?  
            sub_info["distractor"] = [ int(np.squeeze(np.array(i)[i != j])) for i, j in zip(choices, correct_choice) ] # what was the distractor object number?
            sub_info["response"] = [ i [j] for i, j in zip(choices, valid_response) ] # which object with the subject choose?
            sub_info["correct"] = [i == j for i, j in zip(sub_info["response"], sub_info["target"])]
            sub_info["rt"] = valid_rt
            
            unique_objects = np.unique(correct_obj_no) # unique object numbers
            num_objects = len(unique_objects);      # how many objects are there?
            
            # generate confusion matrix
            sub_conf = np.empty((num_objects, num_objects))
            sub_conf[:] = np.nan
            conf_labels = []
            for i in unique_objects:
                for j in unique_objects:
                    cur_idx = [t == i and d == j for t,d in zip(sub_info["target"], sub_info["distractor"])]
                    if np.sum(cur_idx) > 0:
                        sub_conf[i,j] = 1 - np.nanmean(np.array(sub_info["correct"])[cur_idx])
                    if file == file_list[-1]:
                        # labels will be the same for all subjects, so only create for the last one
                        conf_labels.append(labels[i] + "_vs_" + labels[j])
            sub_info["conf"] = sub_conf
            all_data.append(sub_info)
        return(all_data, sub_list, conf_labels)

def confusion_averages(all_data, reject_list=None):
    all_conf = np.array([x["conf"] for x in all_data ])
    all_corr = np.array([np.mean(x["correct"]) for x in all_data ])
    if reject_list is None: reject_list = np.zeros((all_corr.shape[0]), dtype=bool)
    all_corr = all_corr[~reject_list]
    all_conf = all_conf[~reject_list, :, :]
    mean_corr = np.nanmean(all_corr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_conf = np.nanmean(all_conf, 0)
    return(mean_conf, mean_corr, all_conf, all_corr)       

def confusion_list(mean_conf, label_list):
    # get rid of nans and convert to list
    conf_list = list(mean_conf.flatten())

    # combine the two lists and convert confusion to accuracy
    temp = [(1-t[0],t[1]) for t in zip(conf_list, label_list) if not np.isnan(t[0]) ]

    # sort by accuracy and pull apart
    conf_list, label_list = (list(t) for t in zip(*sorted(temp)))

    # this was a variable I used when bebugging
    test = [{t:c} for t,c in zip(conf_list, label_list)]

    return(conf_list, label_list)
    
def confusion_plot(mean_conf, labels, new_labels):
    ## convert the numpy array to a pandas dataframe variable
    conf_df = pd.DataFrame(mean_conf, columns = labels, index = labels)
    # apply the new ordering to the dataframe
    conf_df = conf_df.reindex(columns=new_labels, index=new_labels)
    # now make a figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
    font = {'family' : 'sans-serif',
            'weight' : 'normal',
            'size'   : 12}
    plt.rc('font', **font)
    # first plot a only the nan values, so they can be identified
    sns.heatmap(
        np.where(conf_df.isna(), 0, np.nan),
        ax=ax,
        cbar=False,
        annot=np.full_like(conf_df, "NA", dtype=object),
        fmt="",
        annot_kws={"size": 10, "va": "center_baseline", "color": "black"},
        cmap=ListedColormap(['none']),
        linewidth=0)
    # then overlay the actual values from the non-nan cells
    sns.heatmap(
        conf_df, 
        ax=ax, 
        vmin=0, vmax=.4, 
        cbar=True, cbar_kws={'shrink':.4}, 
        cmap=sns.color_palette("viridis", 100), 
        square=True) 

    # labels the axes
    ax.set_xticklabels(new_labels, rotation = 45, ha="center")
    plt.xlabel("distractor")
    plt.ylabel("target")
    ax.set_yticklabels(new_labels, rotation = 45, va="center")
    ax.tick_params(length=10, width=2)
    plt.tight_layout()
    return plt