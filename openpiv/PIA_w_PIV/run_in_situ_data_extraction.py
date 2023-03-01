
# script to run in_situ_data_extraction.py

# ACW 20 Sept 2022


# =========================================================================================

# Import methods and packages 

import in_situ_data_extraction2 as ide
from importlib import reload
import matplotlib.pyplot as plt

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
analysis_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2023-02-10 12:23:49.046774'

analysis_lookup_file = 'processed_lookup_table.csv'
anlaysis_method = 'A'

classification = 'Copepod'

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

# ---------------------------------------------------------------------------------------------------------------------------------------

# Create the Analysis object 

test = ide.Analysis(rootdir=analysis_folder, lookup_file= analysis_lookup_file, group_method = anlaysis_method,
    oxygen_thresh=oxygen_threshold, time_thresh1=time_threshold_early, time_thresh2=time_threshold_late, depth_thresh=depth_threshold, classifier=classification, 
    save=save_file, output_file=output_file_name)


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

# video level
test.all_group_data[0].group_vids[1]
test.all_group_data[0].group_vids[1].profile
test.all_group_data[0].group_vids[1].paths_of_interest
test.all_group_data[0].group_vids[1].vid_cruise_speed
test.all_group_data[0].group_vids[1].vid_avg_cruise_speed

# path level
test.all_group_data[0].group_vids[1].paths_of_interest[0]
test.all_group_data[0].group_vids[1].paths_of_interest[0].path_length
test.all_group_data[0].group_vids[1].paths_of_interest[0].path_cruise_speeds
test.all_group_data[0].group_vids[1].paths_of_interest[0].path_avg_cruise_speed



# Other tests 
test.video_dic[test.sorted_videos.iloc[1,1]].zoop_paths[6].classification
test.video_dic['1535753947'].zoop_paths[6].classification

test.video_dic['1537773747'].zoop_paths[7].classification

def classification_determination(List, classification, thresh):
    # function to determine if path is cope/amph if above a specified threshold
    # doesnt need to be most frequent
    # IN PROGRESS - has not been tested 
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


