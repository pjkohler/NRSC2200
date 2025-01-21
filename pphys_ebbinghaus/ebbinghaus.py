# import python modules to use for analysis
import numpy as np
import pandas as pd
import os
import subprocess
from IPython.utils import io
import glob
import matplotlib.pyplot as plt
import ast
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# import package for fitting
import psignifit as ps
import psignifit.psigniplot as psp

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

# function for loading all data

def load_complete():
    info_df = pd.read_csv("ebbinghaus_combined.csv")
    return info_df

# function for loading the data into memory

def load_data(data_dir, grab_course = "all", grab_term = "all"):

    if not isinstance(data_dir, list):
        data_dir = [data_dir]

    # define empty lists that will hold the number of subjects, rejected subjects and test subjects
    sub_count = [0]*len(data_dir)    # included datasets
    reject_count = [0]*len(data_dir) # complete datasets, but rejected due to performance
    test_count = [0]*len(data_dir)   # incomplete test datasets

    complete_subs = [] # subjects that produced complete datasets

    prac_n = 16 # number of practice trials
    exp_n = 160 # number of experiment trials

    info_df = pd.DataFrame()

    for e, cur_dir in enumerate(data_dir):
        file_list = glob.glob(cur_dir + "/**/psyc4260*.csv") + glob.glob(cur_dir + "/**/nrsc2200*.csv")
        if grab_course not in ["all", "ALL", "All"]:
            file_list = [x for x in file_list if grab_course.upper() in x.upper() ]
        else:
            print("grabbing data from all courses")
        if grab_term not in ["all", "ALL", "All"]:
            file_list = [x for x in file_list if grab_term.upper() in x.upper() ]
        else:
            print("grabbing data from all terms")
        assert len(file_list) > 0, "No data found for course {} in term {}".format(grab_course.upper(), grab_term.upper())
        file_list.sort()
        exp_subs = [] # list to hold the subjects in this experiment
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
                survey_resp = sub_data[sub_data["trial_type"]=="survey-html-form"].loc[0]["response"]
                survey_resp = survey_resp.replace(':"}',':""}')
                sub_info = ast.literal_eval(survey_resp)
            except:
                sub_info = {}
            # see if id was stored
            if 'p_id' in sub_info.keys():
                # if participant ID was recorded, use it to get rid of data with test subject ID 214984
                # this is the ID associated with URPP's sample link
                try:    
                    sub_id = sub_info["p_id"]
                except:
                    sub_id = "nan"
                del sub_info['p_id']
                if sub_id in ["214984", "603921b616b3893be91674d1"]:
                    test_count[e] = test_count[e] + 1
                    continue
            else:
                sub_id = "nan"
            # do quality control on the data
            if sub_data.shape[1] < 10:
                print(e, sub_id, "incomplete file, shape:{0}x{1}".format(sub_data.shape[0],sub_data.shape[1]))
                test_count[e] = test_count[e] + 1 # record test subject to the test_count, by adding 1 at the relevant position
                continue
            
            # now skip if this subjects data is already in the set
            if sub_id in exp_subs:
                continue
            
            # add participant to list of subjects for this experiment
            exp_subs.append(sub_id)
            
            # start populating sub_info dict: 
            sub_info["experiment"] = e
            sub_info["ID"] = sub_id
            
            # get course and timing for this subject
            try:
                course = file.split('-')[0].split('/')[-1].upper()
                dt = datetime.strptime(sub_data["recorded_at"][0], '%Y-%m-%d %H:%M:%S')
                if dt.month < 5:
                    sub_info["course"] = "{}_W{}".format(course, dt.year)
                elif dt.month > 8:
                    sub_info["course"] = "{}_F{}".format(course, dt.year)
                else:
                    sub_info["course"] = "{}_S{}".format(course, dt.year)
            except:
                sub_info["course"] = file.split('/')[3].upper()

            # Cognition.run specific error: replace strings with logicals
            sub_data = sub_data.replace("true", True)
            sub_data = sub_data.replace("false", False)
            
            pphys_data = sub_data[sub_data["trial_type"] == "psychophysics"]
            pphys_data = pphys_data.replace(np.nan, "")
            pphys_data = pphys_data.replace('"', "")
            pphys_data = pphys_data[pphys_data.test_inner_radius != ""]
            pphys_data = pphys_data.astype(
                {"test_inner_radius": 'int', "test_outer_radius": 'int', "ref_inner_radius": 'int', "ref_outer_radius": 'int' })
            
            test_inner_sizes = [x for x in pphys_data["test_inner_radius"].unique()]
            test_inner_sizes.sort()
            
            test_outer_sizes = [x for x in pphys_data["test_outer_radius"].unique()]
            test_outer_sizes.sort()
            
            ref_outer_size = [x for x in pphys_data["ref_outer_radius"].unique()][0]
            ref_inner_size = [x for x in pphys_data["ref_inner_radius"].unique()][0]
            
            rt_reject = False
            acc_reject = False
            for size in test_inner_sizes:
                for o in test_outer_sizes:
                    cur_data = pphys_data[(pphys_data["test_inner_radius"]==size) & (pphys_data["test_outer_radius"]==o)]
                    # only include trials with RTs less than 5 secs
                    cur_data = cur_data[cur_data["rt"] < 3000]
                    if len(cur_data) < 10:
                        rt_reject = True
                    if o == ref_outer_size:
                        sub_info["rt-same-"+str(size)] = cur_data["rt"].mean()
                        temp_bigger = cur_data["test_bigger"]
                        sub_info["testbigger-same-" + str(size) ] = np.nansum(temp_bigger)/len(temp_bigger)
                        sub_info["ntrials-same-" + str(size) ] = len(temp_bigger)
                        if size == test_inner_sizes[0]:
                            if np.nansum(temp_bigger)/len(temp_bigger) > 0.2:
                                acc_reject = True
                        if size == test_inner_sizes[-1]:
                            if np.nansum(temp_bigger)/len(temp_bigger) < 0.8:
                                acc_reject = True
                    else:
                        sub_info["rt-small-"+str(size)] = cur_data["rt"].mean()
                        temp_bigger = cur_data["test_bigger"]
                        sub_info["testbigger-small-" + str(size) ] = np.nansum(temp_bigger)/len(temp_bigger)
                        sub_info["ntrials-small-" + str(size) ] = len(temp_bigger)
            if acc_reject:
                print("Excluding {} : {}, chance performance".format(sub_info["course"], sub_id))
                continue 
            elif rt_reject:
                print("Excluding {} : {}, less than 10 trials with RT < 3 secs".format(sub_info["course"], sub_id))
                continue
            sub_count[e] = sub_count[e]+1
            info_df = pd.concat([info_df, pd.json_normalize(sub_info)], sort=False)

            # add to dataframe of complete subjects 
            complete_subs.append(exp_subs)
            
    cols = info_df.columns.tolist()
    cols.sort()
    cols.insert(0, cols.pop(cols.index("experiment")))
    if "age" in cols:
        cols.insert(2, cols.pop(cols.index("age")))
        cols.insert(3, cols.pop(cols.index("handedness")))
        cols.insert(4, cols.pop(cols.index("sex")))
        cols.insert(5, cols.pop(cols.index("other_sex")))
    info_df = info_df[cols]
    info_df = info_df.sort_values(by = ["experiment", "course", "ID"] )
    info_df["count"] = range(0,len(info_df))
    info_df = info_df.set_index("count")

    return info_df

# functions for fitting psychometric functions

def fit_ps(stim_params, fit_data, n_trials, options=dict()):
    fit_data = np.vstack([np.array(stim_params), np.array(fit_data)*np.array(n_trials), np.array(n_trials)])
    if not options:
        options = dict();   # initialize as an empty dictionary
        options['sigmoid'] = 'norm';   # choose a cumulative Gauss as the sigmoid  
        options['experiment_type']     = 'equal asymptote';   # choose 2-AFC as the experiment type  
                                   # this sets the guessing rate to .5 (fixed) and  
                                   # fits the rest of the parameters
    with io.capture_output() as captured:
        temp_params = ps.psignifit(fit_data.transpose(), **options)
    threshold = temp_params.parameter_estimate['threshold']
    slope = temp_params.slope(0.5)
    return temp_params, threshold, slope

def combine_data(new_params, all_params=None, all_options=[]):
    new_params['options']['timestamp'] = new_params['timestamp']
    all_options.append(new_params['options'])
    if not all_params:
        all_params = dict()
        new_data = True
    else:
        new_data = False
    for key, value in new_params.items():
        if isinstance(value, float):
            if new_data:
                all_params[key] = [value] # turn into list
            else:
                all_params[key].append(value)
        elif isinstance(value, np.ndarray):
            if new_data:
                all_params[key] = value[..., np.newaxis]
            else:
                all_params[key] = np.concatenate((all_params[key], value[..., np.newaxis]), -1)
        if isinstance(value, list):
            if new_data:
                all_params[key] = value
                for c, data in enumerate(value):
                    all_params[key][c] = data[..., np.newaxis]
            else:
                for c, data in enumerate(value):
                    all_params[key][c] = np.concatenate((all_params[key][c], data[..., np.newaxis]), -1)
    return all_params, all_options

def fit_data(info_df, data_type="same", lightweight=True):
    pse_slope = [ ]
    fit_params = [ ]
    print('Fitting psychometric functions for "{}" condition ... '.format(data_type), end='')
    test_inner_sizes = [int(x.split("-")[-1])for x in list(info_df.columns.values) if x.startswith("testbigger-{0}-".format(data_type)) ]
    for index, row in info_df.iterrows():
        # same inducers
        fit_data = [ row[x] for x in info_df.columns if x.startswith("testbigger-{0}-".format(data_type)) ]
        n_trials = [ row[x] for x in info_df.columns if x.startswith("ntrials-{0}-".format(data_type)) ]
        temp_params, threshold, slope = fit_ps(test_inner_sizes, fit_data, n_trials)
        pse_slope.append((threshold, slope))
        fit_params.append(temp_params)
        #if index == 0:
        #    fit_params, fit_options = combine_data(temp_params)
        #else:
        #    fit_params, fit_options = combine_data(temp_params, fit_params, fit_options)
        temp_params = None 
    # run fit on averages for illustration purposes
    t_f = np.array([info_df[x] for x in info_df.columns if x.startswith("testbigger-{0}-".format("small"))])
    t_n = np.array([info_df[x] for x in info_df.columns if x.startswith("ntrials-{0}-".format("small"))])
    fit_data = [int(x) for x in list(np.sum(t_n * t_f, 1))]
    n_trials = [int(x) for x in list(np.sum(t_n, 1))]
    temp_params, threshold, slope = fit_ps(test_inner_sizes, fit_data, n_trials)
    pse_slope.append((threshold, slope))
    #fit_params, fit_options = combine_data(temp_params, fit_params, fit_options)
    fit_params.append(temp_params)
    temp_params = None
    if lightweight:
        fit_params = []
    # assign pses to info_df
    info_df["pse-{0}".format(data_type)] = [ x[0] for x in pse_slope[0:-1] ]
    info_df["pse-{0}-avefit".format(data_type)] = [pse_slope[-1][0]] * info_df.shape[0]
    info_df["slope-{0}".format(data_type)] = [ x[1] for x in pse_slope[0:-1] ]
    info_df["slope-{0}-avefit".format(data_type)] = [pse_slope[-1][1]] * info_df.shape[0]
    print('finished!'.format(data_type))
    # save data to a csv
    info_df.to_csv("ebbinghaus_combined.csv")

    return info_df, fit_params

# function for doing plotting

def plot_ps(info_df, fit_params, participant, annotate=False, plot_function=False):
    # plot data from two experiments
    condition_list = ["same", "small"]
    
    if not participant in ["average", "averages", "mean", "means"]:
        if participant in info_df['ID'].to_list():
            cur_df = info_df.loc[info_df['ID'] == participant]
            idx = info_df[info_df['ID'] == participant].index[0]
            cur_params = [ x[idx] for x in fit_params ] 
            # use pse and slope from individual subject
            cur_pse = [cur_df["pse-{0}".format(c)].to_list()[0] for c in condition_list ]
            cur_slope = [cur_df["slope-{0}".format(c)].to_list()[0] for c in condition_list ]
        else:
            print("Unknown participant! Try again!")
            return
    else:
        cur_df = info_df
        cur_params = [ x[-1] for x in fit_params ]
        # use pse and slope from average fit
        cur_pse = [cur_df["pse-{0}-avefit".format(c)].to_list()[0] for c in condition_list ]
        cur_slope = [cur_df["slope-{0}-avefit".format(c)].to_list()[0] for c in condition_list ]

    ps_kwargs = dict(plot_data=False, plot_parameter=False)
    eb_kwargs = dict(linestyle='none', mfc='none', ms=8, clip_on=False)

    fig, ax = plt.subplots(1,2, figsize=[15,5])
    for t, data_type in enumerate(["rt", "testbigger"]):
        for c, cond in enumerate(condition_list):
            test_inner_sizes = [int(x.split("-")[-1])for x in list(info_df.columns.values) if x.startswith("testbigger-{0}-".format(cond)) ]
            means = [cur_df["{0}-{1}-{2}".format(data_type, cond, str(x))].mean() for x in test_inner_sizes]
            stddev = [cur_df["{0}-{1}-{2}".format(data_type, cond, str(x))].std() for x in test_inner_sizes]
            n = [cur_df["{0}-{1}-{2}".format(data_type, cond, str(x))].count() for x in test_inner_sizes]
            stderr = [i / np.sqrt(j) for i, j in zip(stddev, n)]
            ci = [x * 1.96 for x in stderr]
            if cond == 'same':
                eb_kwargs["marker"]='o'
            else:
                eb_kwargs["marker"]='s'
            if all([x == 1 for x in n]):
                e_h = ax[t].plot(test_inner_sizes, means, label=condition_list[c]+ " inducers", **eb_kwargs)
            else:
                e_h = ax[t].errorbar(test_inner_sizes, means, yerr=ci, label=condition_list[c]+ " inducers", **eb_kwargs)

            ax[t].set_xticks(test_inner_sizes)

            if data_type == 'rt':
                ax[t].set_ylim([0,5000])
            else:
                ax[t].set_ylim([0,1])
            if "testbigger" in data_type:
                title_str = 'Responses'
                ylabel_str = "test > ref (proportion)"
                many_xx = np.linspace(20, 30)
                if plot_function:
                    psp.plot_psychometric_function(cur_params[c], ax=ax[t],line_color=e_h[0].get_color(), **ps_kwargs)
                ax[t].plot((10, 40), (0.5, 0.5), 'k:')
                
                if annotate:
                    if cond == "same":
                        ax[t].text(26, 0.40, "PSE = {:.2f}, slope = {:.2f}".format(cur_pse[c], cur_slope[c]), color=e_h[0].get_color())
                    else:
                        ax[t].text(26, 0.45, "PSE = {:.2f}, slope = {:.2f}".format(cur_pse[c], cur_slope[c]), color=e_h[0].get_color())
            else:
                title_str = 'Reaction Time'
                ylabel_str = "RT (ms)"
            ax[t].set_xlim([19,31])  

            ax[t].set_xlabel("inner test size (ref: 25)", fontsize=12)
            ax[t].set_ylabel(ylabel_str)
            ax[t].title.set_text(title_str)
            ax[t].legend(frameon=False)
            ax[t].spines['top'].set_visible(False)
            ax[t].spines['right'].set_visible(False)
            for item in ([ax[t].title] + [ax[t].xaxis.label, ax[t].yaxis.label] +
                        ax[t].get_xticklabels() + ax[t].get_yticklabels() + 
                        ax[t].get_legend().get_texts()):
                item.set_fontsize(12)
                item.set_fontname("DejaVu Sans")
            