
# script to run in_situ_data_extraction.py

# ACW 20 Sept 2022

# To execute: 
    # cd Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV
    # python3.8
    # copy and paste code into terminal 

# =========================================================================================

# Import methods and packages 

#import in_situ_data_extraction2 as ide
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import statistics
import datetime
import os

# =========================================================================================

# Read Me:

# Overview of grouping methods:
    # A -- hypoxic/normoxic
    # B -- hypoxic/normoxic, deep/shallow, AM/PM
    # C -- hypoxic/normoxic, deep/shallow
    # D -- hypoxic/normoxic, AM/PM

# =========================================================================================

# Define parameters:

# SELECT SWIMMING ANALYSIS
import in_situ_data_extraction_copepod2 as ide
#import in_situ_data_extraction_amphipod as ide

analysis_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2023-03-24 15:58:07.367266'

analysis_lookup_file = 'processed_lookup_table.csv'
anlaysis_method = 'A'

classification = 'Copepod'
#classification = 'Amphipod'

oxygen_threshold = 2
time_threshold_early = 7
time_threshold_late = 19
depth_threshold = 50

save_file = True
#save_file = False

# Hidden Markov Model parameters
HMM_start_points = 1000
HMM_niter = 1000

# ---------------------------------------------------------------------------------------------------------------------------------------

# Generate informative output file name

if anlaysis_method == 'A':
    output_file_name = str('post_processed_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '.csv')
elif anlaysis_method == 'B':
    output_file_name = str('post_processed_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '_time' + str(time_threshold_early) + '_time' + str(time_threshold_late) + '_depth' + str(depth_threshold) + '.csv')
elif anlaysis_method == 'C':
    output_file_name = str('post_processed_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '_depth' + str(depth_threshold) + '.csv')
elif anlaysis_method == 'D':
    output_file_name = str('post_processed_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '_time' + str(time_threshold_early) + '_time' + str(time_threshold_late) + '.csv')
else:
    print('Grouping method is not definited')

if anlaysis_method == 'A':
    metadata_file_name = str('metadata_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '.csv')
elif anlaysis_method == 'B':
    metadata_file_name = str('metadata_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '_time' + str(time_threshold_early) + '_time' + str(time_threshold_late) + '_depth' + str(depth_threshold) + '.csv')
elif anlaysis_method == 'C':
    metadata_file_name = str('metadata_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '_depth' + str(depth_threshold) + '.csv')
elif anlaysis_method == 'D':
    metadata_file_name = str('metadata_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '_time' + str(time_threshold_early) + '_time' + str(time_threshold_late) + '.csv')
else:
    print('Grouping method is not definited')

if anlaysis_method == 'A':
    markov_file_name = str('markov_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '.csv')
elif anlaysis_method == 'B':
    markov_file_name = str('markov_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '_time' + str(time_threshold_early) + '_time' + str(time_threshold_late) + '_depth' + str(depth_threshold) + '.csv')
elif anlaysis_method == 'C':
    markov_file_name = str('markov_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '_depth' + str(depth_threshold) + '.csv')
elif anlaysis_method == 'D':
    markov_file_name = str('markov_mtd'+ anlaysis_method +'_' + classification +'_oxyg' + str(oxygen_threshold) + '_time' + str(time_threshold_early) + '_time' + str(time_threshold_late) + '.csv')
else:
    print('Grouping method is not definited')

# ---------------------------------------------------------------------------------------------------------------------------------------

# Create the Analysis object 
#reload(ide)

final_extraction = ide.Analysis(rootdir=analysis_folder, lookup_file= analysis_lookup_file, group_method = anlaysis_method,
    oxygen_thresh=oxygen_threshold, time_thresh1=time_threshold_early, time_thresh2=time_threshold_late, depth_thresh=depth_threshold, classifier=classification,
    HMM_start=HMM_start_points, HMM_n=HMM_niter, 
    save=save_file, output_file=output_file_name, metadata_file=metadata_file_name, markov_file=markov_file_name)
    
# AFTER IT FINISHES --- should automate this 
# Pickle entire analysis object after running it
pickle_name = str('Final_Extraction_Analysis_meth'+ anlaysis_method +'_' + classification +'_starts' + str(HMM_start_points) + '_niter' + str(HMM_niter) + '.pickle')
pickle_path = os.path.join(analysis_folder, pickle_name)
pickle_file = open(pickle_path, 'wb')
pickle.dump(final_extraction, pickle_file)
pickle_file.close()
print("Pickeled ", pickle_file)


# =========================================================================================
# =========================================================================================
# =========================================================================================
# =========================================================================================
# Output! 

final_extraction.video_dic['1537773747'].zoop_paths[7].speed_raw

test.all_group_data[0].HMM_mean_all
test.all_group_data[0].HMM_mean_size_0_1
test.all_group_data[0].HMM_mean_size_1_2
test.all_group_data[0].HMM_mean_size_2_3
test.all_group_data[0].HMM_mean_size_3_4
test.all_group_data[0].HMM_mean_size_4_10

test.all_group_data[1].HMM_mean_all
test.all_group_data[1].HMM_mean_size_0_1
test.all_group_data[1].HMM_mean_size_1_2
test.all_group_data[1].HMM_mean_size_2_3
test.all_group_data[1].HMM_mean_size_3_4
test.all_group_data[1].HMM_mean_size_4_10

# new amphipod analysis 
test.all_group_data[0].group_vids[2].paths_of_interest[0].path_slow_speeds
test.all_group_data[0].group_vids[2].paths_of_interest[0].path_fast_speeds
test.all_group_data[0].group_vids[2].paths_of_interest[0].speed_states
test.all_group_data[0].group_vids[2].paths_of_interest[0].trans_matrix

# all data
test.all_group_data
test.video_dic
test.sorted_videos
test.lookup_table
test.df

# group level
test.all_group_data[0].group
test.all_group_data[0].paths_per_vid
test.all_group_data[0].group_vids
test.all_group_data[0].group_cruise_speed
test.all_group_data[0].group_avg_cruise_speed
test.all_group_data[0].group_speed_states
test.all_group_data[0].trans_matrix
test.all_group_data[0].trans_matrix_prob

sum(test.all_group_data[0].trans_matrix_prob[0])


# video level
test.all_group_data[0].group_vids[1]
test.all_group_data[0].group_vids[1].profile
test.all_group_data[0].group_vids[1].paths_of_interest
test.all_group_data[0].group_vids[1].vid_cruise_speed
test.all_group_data[0].group_vids[1].vid_avg_cruise_speed

# path level
test.all_group_data[0].group_vids[3].paths_of_interest[0]
test.all_group_data[0].group_vids[1].paths_of_interest[0].path_length
test.all_group_data[0].group_vids[1].paths_of_interest[0].path_cruise_speeds
test.all_group_data[0].group_vids[1].paths_of_interest[0].path_avg_cruise_speed


len(test.sorted_videos[test.sorted_videos[0] == 'hypoxic'])     # 80
len(test.sorted_videos[test.sorted_videos[0] == 'normoxic'])    # 107 




# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

# Experiemnts 


# Marvok stats test ------------------------------------------------
# HMM Learn model 
        # I think getting to work
from hmmlearn.hmm import GaussianHMM
from matplotlib import cm, pyplot as plt
from sklearn.utils import check_random_state

# a) combine three swimming paths to test multiple sequences 
X1 = test.all_group_data[0].group_vids[2].paths_of_interest[0].all_speeds
X2 = test.all_group_data[0].group_vids[2].paths_of_interest[1].all_speeds
X3 = test.all_group_data[0].group_vids[2].paths_of_interest[6].all_speeds
X1 = [[i] for i in X1]
X2 = [[i] for i in X2]
X3 = [[i] for i in X3]

X = np.concatenate([X1, X2, X3])
L = [len(X1), len(X2), len(X3)]
F = np.array(range(len(X)))

# b)  Using AIC values to determine best model -- number of componments
        # https://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_gaussian_model_selection.html#sphx-glr-auto-examples-plot-gaussian-model-selection-py 

rs = check_random_state(546)

start_point = 10          # number of starting points
n_iter = 10

aic = []
bic = []
lls = []
models = []

ns = [2, 3, 4, 5, 6]

for n in ns:
    best_ll = None
    best_model = None
    for i in range(start_point):                                                 
        h = GaussianHMM(n, n_iter=n_iter, tol=1e-4, random_state=rs)
        h.fit(X,L)
        score = h.score(X)
        #if not best_ll or best_ll < best_ll:                # typo 
        if not best_ll or best_ll < score:                  # saves LARGER value
            best_ll = score
            best_model = h
    aic.append(best_model.aic(X))
    bic.append(best_model.bic(X))
    lls.append(best_model.score(X))
    models.append(best_model)

fig, ax = plt.subplots()
ln1 = ax.plot(ns, aic, label="AIC", color="blue", marker="o")
ln2 = ax.plot(ns, bic, label="BIC", color="green", marker="o")
ax2 = ax.twinx()
ln3 = ax2.plot(ns, lls, label="LL", color="orange", marker="o")
ax.legend(handles=ax.lines + ax2.lines)
ax.set_title("Using AIC/BIC for Model Selection")
ax.set_ylabel("Criterion Value (lower is better)")
ax2.set_ylabel("LL (higher is better)")
ax.set_xlabel("Number of HMM Components")
#fig.tight_layout()

# c) determine means, variance, and prob matrix: 
    # https://hmmlearn.readthedocs.io/en/0.2.0/auto_examples/plot_hmm_stock_analysis.html 

rs = check_random_state(546)            # this is usually for debugging purposes -- dont need to reset it before running model

model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=1000, random_state=rs).fit(X, L)      # can read in saved model from previous step 

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(X, L)

print("Transition matrix")
print(model.transmat_)

# this looks promising! 
print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

# dont know what this is supposed to show me, but plotting data
fig, axs = plt.subplots(model.n_components, sharex=True, sharey=True)
colours = cm.rainbow(np.linspace(0, 1, model.n_components))
for i, (ax, colour) in enumerate(zip(axs, colours)):
    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    #ax.plot_date(dates[mask], close_v[mask], ".-", c=colour)
    ax.plot_date(F[mask], X[mask], ".-", c=colour)
    ax.set_title("{0}th hidden state".format(i))


# GROUP transition states ----------------------------------
pd.set_option('display.max_rows', None)

paths = []
path1 = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
path2 = ['C', 'C', 'C', 'J', 'J', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']
path3 = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'D']
paths.append(path1)
paths.append(path2)
paths.append(path3)

df_total = pd.DataFrame()
for p in paths:
    df = pd.DataFrame(p)
    df['shift'] = df[0].shift(-1)
    # add a count column (for group by function)
    df['count'] = 1
    #print(df)
    df_total = pd.concat([df_total, df])

trans_mat = df_total.groupby([0, 'shift']).count().unstack().fillna(0)
trans_mat_fin = trans_mat.div(trans_mat.sum(axis=1), axis=0).values

# Binned by size 
# c("0-1","1-2","2-3","3-4","4-10"))
for v in self.group_vids:
    for p in v.paths_of_interest:
        if p.path_avg_length <= 1:
            self.group_speed_states_1.append(p.speed_states)
        elif p.path_avg_length > 1 and p.path_avg_length <=2:
            self.group_speed_states_2.append(p.speed_states)
        elif p.path_avg_length > 2 and p.path_avg_length <=3:
            self.group_speed_states_3.append(p.speed_states)
        elif p.path_avg_length > 3 and p.path_avg_length <=4:
            self.group_speed_states_4.append(p.speed_states
        elif p.path_avg_length > 4 and p.path_avg_length <=10:
            self.group_speed_states_10.append(p.speed_states)

df_total = pd.DataFrame()
for p in self.group_speed_states:
    df = pd.DataFrame(p)
    df['shift'] = df[0].shift(-1)
    df['count'] = 1
    df_total = pd.concat([df_total, df])
self.trans_matrix = df_total.groupby([0, 'shift']).count().unstack().fillna(0)
self.trans_matrix_prob = self.trans_matrix.div(self.trans_matrix.sum(axis=1), axis=0).values


# Histogram of all speeds ----------------------------------
speeds = []
for i in test.all_group_data[0].group_vids:
    #print(i)
    for j in i.paths_of_interest:
        speeds.append(j.all_speeds)

flat_list = [item for sublist in speeds for item in sublist]

statistics.mean(flat_list)

plt.hist(flat_list, bins=100)

# Classification Method ----------------------------------
test.video_dic[test.sorted_videos.iloc[1,1]].zoop_paths[6].classification
test.video_dic['1535753947'].zoop_paths[6].classification

test.video_dic['1537773747'].zoop_paths[7].classification

def classification_determination(List, classification, thresh):
    # function to determine if path is cope/amph if above a specified threshold
    # doesnt need to be most frequent
    
    # NEED: dont include snow in fraction thresholds 
    #   if its 30% copepod and 70% amphipod you dont want to call it a copepod
    #   generally need to rethink how I filter 
    
    print(len(List))
    count = 0
    for i in List:
        if i == classification:
            count = count + 1
    print(count)
    print(count/len(List))
    if count/len(List) > thresh:
        print('true')
        return classification

class_test = classification_determination(List = test.video_dic['1537773747'].zoop_paths[10].classification, classification= 'Copepod', thresh=0.25)












# Copepod size
test.video_dic['1537773747'].zoop_paths[6].classification
test.video_dic['1537773747'].zoop_paths[6].length
test.video_dic['1537773747'].zoop_paths[6].area

test.all_group_data[0].group_cruise_speed

test.all_group_data[0].group_vids[2].paths_of_interest[0].path_classifications
test.all_group_data[0].group_vids[2].paths_of_interest[0].path_lengths
test.all_group_data[0].group_vids[2].paths_of_interest[0].path_areas

test.all_group_data[0].group_vids[2].paths_of_interest[0].path_avg_length
test.all_group_data[0].group_vids[2].paths_of_interest[0].path_avg_area


group_test = test.all_group_data[0]
video_test = test.all_group_data[0].group_vids[3]
path_test = test.all_group_data[0].group_vids[3].paths_of_interest[0]

group_test.group
video_test.profile
test.lookup_table[test.lookup_table[:,0] == video_test.profile,1][0]
test.lookup_table[test.lookup_table[:,0] == video_test.profile,5][0]
test.lookup_table[test.lookup_table[:,0] == video_test.profile,6][0]
test.lookup_table[test.lookup_table[:,0] == video_test.profile,7][0]
path_test.path_ID # trouble
path_test.path_length
path_test.path_max_area # trouble
path_test.path_max_length # trouble
path_test.path_avg_cruise_speed
path_test.path_jumps
path_test.path_avg_jump_speed # trouble


