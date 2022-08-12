
# a script to recreate the lookup table without having to reprocess all the pickled video profiles
# this is only necessary if you have already run a long analysis (ideally you would just modify the batch_run_PIA_w_PIV.py script ~line 138)

# ACW August 2022

# ========================================================================

import os 
import numpy as np
import pickle
import statistics
import pandas as pd


# ========================================================================

# directory of pickled videos

rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-04 14:11:00.316190'

# ========================================================================

# read in pickled videos 

video_dic = {}

for file in os.listdir(rootdir):
    if file.endswith('.pickle'):
        print(str(file)[0:10])
        pickle_file = open(os.path.join(rootdir,file),"rb")
        print(pickle_file)
        video_dic[str(file)[0:10]] =  pickle.load(pickle_file)
        pickle_file.close()

keys_list = list(video_dic)

# ========================================================================

# recreate the lookup table with new variables

analysis_objs_created = []

for i in range(len(keys_list)):
    
    video = video_dic[keys_list[i]]
    
    # Data Variables:
    profile = keys_list[i]
    datetime = video.profile
    if len(video.zoop_paths) > 0:
        approx_frame_rate = 1 / statistics.mean(video.zoop_paths[0].delta_time_corrected)
    else:
        approx_frame_rate = 'NaN'
    total_frames = len(video.full_flowfield.tif_list)
    num_paths = len(video.zoop_paths)
    depth = video.depth_avg
    oxygen = video.oxygen_mgL_avg
    temp = video.temp_avg
    nearest_ctd = video.nearest_earlier_cast
    nearest_ctd_offset = video.profile - pd.Timestamp(video.nearest_earlier_cast, tz='US/Pacific')

    analysis_info = [profile, datetime, approx_frame_rate, total_frames, num_paths, depth, oxygen, temp,  nearest_ctd, nearest_ctd_offset]
    analysis_objs_created.append(analysis_info)

# ========================================================================

# save a new file in same directory 

processed_lookup = np.array(analysis_objs_created)
processed_lookup_file = 'processed_lookup_table_modified.csv'
processed_lookup_path = os.path.join(rootdir, processed_lookup_file)
np.savetxt(processed_lookup_path, processed_lookup, delimiter=',', fmt='%s', header='profile, datetime, approx_frame_rate, total_frames, num_paths, depth, oxygen, temp,  nearest_ctd, nearest_ctd_offset')



