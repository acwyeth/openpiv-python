
# a script to read in pickled video analysis objects using a look up table, unpickle, and organize final data

# ACW

# 15 July 2022

# Execute:
    # cd Wyeth2/GIT_repos_insitu/GIT_in_situ_motion

# ==================================================================

import os 
import numpy as np
import pickle
import statistics
from collections import Counter


# ==================================================================

# directories and filenames

rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-04 14:11:00.316190'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-03 12:24:34.550180'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-01 18:01:18.368946'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-07-15 13:39:06.936432'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-07-18 14:27:30.607462'

lookup_file = 'processed_lookup_table.csv'

# ==================================================================

# Read in processed videos and create a dictionary 

lookup_table = np.genfromtxt(os.path.join(rootdir,lookup_file), dtype = str, delimiter=',', skip_header=0)

video_dic = {}

for file in os.listdir(rootdir):
    if file.endswith('.pickle'):
        print(str(file)[0:10])
        pickle_file = open(os.path.join(rootdir,file),"rb")
        print(pickle_file)
        video_dic[str(file)[0:10]] =  pickle.load(pickle_file)
        pickle_file.close()

keys_list = list(video_dic)

# ==================================================================

# define methods 

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

# ==================================================================

# Analysis #1
    # Split paths into hypoxic/normoxic and calculate mean speed across all videos 
    
# Notes:
    # Only looking at paths where majority of IDs is 'Copepod'
    # Skipping paths with flowfield NaNs for now (need to look into this more...)

hypoxic_paths = []
normoxic_paths = []
broken_flowfield = []

for video in lookup_table:
    #print(video)
    if (float(video[3])) <= 2:                                         # set oxygen threshold
        for path in video_dic[video[0]].zoop_paths:
            if not np.isnan(path.x_flow_smoothed).any():                # skip paths with broken smoothing (for now)
                #print(path.x_flow_smoothed)
                if most_frequent(path.classification) == 'Copepod':     # only grab paths that are mostly IDed as copepods 
                    #print(path)
                    hypoxic_paths.append(path)
            else:
                broken_flowfield.append(video[0])
    else:
        for path in video_dic[video[0]].zoop_paths:
            if not np.isnan(path.x_flow_smoothed).any():                # skip paths with broken smoothing (for now)
                if most_frequent(path.classification) == 'Copepod':
                    normoxic_paths.append(path)
            else:
                broken_flowfield.append(video[0])

# Calculate some simple swimming statistics 

hypx_path_avg_speed = []
norm_path_avg_speed = []

for path in hypoxic_paths:
        #print(path.speed)
        path_avg_speed = statistics.mean(path.speed)
        hypx_path_avg_speed.append(path_avg_speed)
hypox_speed_avg = statistics.mean(hypx_path_avg_speed)
hypox_speed_sd = statistics.stdev(hypx_path_avg_speed)

for path in normoxic_paths:
        path_avg_speed = statistics.mean(path.speed)
        norm_path_avg_speed.append(path_avg_speed)
norm_speed_avg = statistics.mean(norm_path_avg_speed)
norm_speed_sd = statistics.stdev(norm_path_avg_speed)


# Output: 
print("Hypoxic: ", hypox_speed_avg, " +/- ", hypox_speed_sd)
print("Normoxic: ", norm_speed_avg, " +/- ", norm_speed_sd)
print("Broken Flowfields: ", np.unique(broken_flowfield))





# ==================================================================

# Misc. code 

video_dic[keys_list[0]].profile
video_dic[keys_list[1]].zoop_paths
video_dic[keys_list[1]].zoop_paths[1].x_motion



