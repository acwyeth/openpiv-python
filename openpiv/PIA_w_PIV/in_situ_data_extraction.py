
# a script to read in pickled video analysis objects using a look up table, unpickle, and organize final data

# ACW

# 15 July 2022

# Execute:
    # cd Wyeth2/GIT_repos_insitu/GIT_in_situ_motion

# ==================================================================

import os 
import numpy as np
import pickle

# ==================================================================

rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-07-15 13:39:06.936432'

lookup_file = 'processed_lookup_table.csv'

# ==================================================================

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


video_dic[keys_list[0]].profile
video_dic[keys_list[1]].zoop_paths
video_dic[keys_list[1]].zoop_paths[1].x_motion      # I think this read in backwards? the 999 should be the last frame










