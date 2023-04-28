
# script to run in_situ_data_extraction.py

# ACW 20 Sept 2022


# =========================================================================================

# Import methods and packages 

import in_situ_data_extraction2 as ide
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
import statistics


reload(ide)

# =========================================================================================

# Read Me:

# Overview of grouping methods:
    # A -- hypoxic/normoxic
    # B -- hypoxic/normoxic, deep/shallow, AM/PM
    # C -- hypoxic/normoxic, deep/shallow
    # D -- hypoxic/normoxic, AM/PM

# =========================================================================================

# Define parameters:

#analysis_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-09-20 16:26:10.449537'
#analysis_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2023-02-10 12:23:49.046774'
#analysis_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2023-03-09 10:08:09.300375'
#analysis_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2023-03-14 09:37:56.373538'
#analysis_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output_tests/2023-03-15 12:06:08.952701'
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

test = ide.Analysis(rootdir=analysis_folder, lookup_file= analysis_lookup_file, group_method = anlaysis_method,
    oxygen_thresh=oxygen_threshold, time_thresh1=time_threshold_early, time_thresh2=time_threshold_late, depth_thresh=depth_threshold, classifier=classification, 
    save=save_file, output_file=output_file_name, metadata_file=metadata_file_name, markov_file=markov_file_name)


# =========================================================================================
# Experiment with charts in python

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('Copepods')
ax1.bar(test.df['Group'],test.df['Avg Cruise Speed'])
ax1.set(ylabel="Avg Cruise Speed (mm/s)")
ax2.bar(test.df['Group'],test.df['Avg Jumps per Frame'])
ax2.set(ylabel="Avg Jumps per Frame")

# =========================================================================================

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

# GROUP transition states
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



# Transition states
path = ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'D', 'D', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C']

df = pd.DataFrame(path)

# create a new column with data shifted one space
df['shift'] = df[0].shift(-1)
# add a count column (for group by function)
df['count'] = 1

# create dummy rows for each transition so the matrix is complete even if there isn't every transition in each path
df.loc[len(df.index)] = ['C', 'C', 0] 
df.loc[len(df.index)] = ['C', 'D', 0] 
df.loc[len(df.index)] = ['C', 'J', 0] 
df.loc[len(df.index)] = ['J', 'C', 0] 
df.loc[len(df.index)] = ['J', 'D', 0] 
df.loc[len(df.index)] = ['J', 'J', 0] 
df.loc[len(df.index)] = ['D', 'C', 0] 
df.loc[len(df.index)] = ['D', 'D', 0] 
df.loc[len(df.index)] = ['D', 'J', 0] 

# groupby and then unstack, fill the zeros
trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)

# remove each of the dummy counts
trans_mat = trans_mat - 1 

# normalise by occurences and save values to get transition matrix
trans_mat_fin = trans_mat.div(trans_mat.sum(axis=1), axis=0).values

trans_mat_fin[np.isnan(trans_mat_fin)] = 0







# Histogram of all speeds
speeds = []
for i in test.all_group_data[0].group_vids:
    #print(i)
    for j in i.paths_of_interest:
        speeds.append(j.all_speeds)

flat_list = [item for sublist in speeds for item in sublist]

statistics.mean(flat_list)

plt.hist(flat_list, bins=100)







# Other tests 
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


