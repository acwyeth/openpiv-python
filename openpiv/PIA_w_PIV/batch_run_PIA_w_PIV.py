
# Script to batch process and save analysis objects for numerous videos 

# ACW 11 July 2022

# Notes:
 
# Execute:
    # 1) cd dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV
    # 2) python3.8 -m batch_run_PIA_w_PIV
    # 3) script will prompt you to enter a note (hit enter TWICE) and it will save an new timestamped folder in the analysis output directory

# =====================================================================================================

from importlib import reload
import sys
import os
from pathlib import Path
import numpy as np
import time
import datetime
import pickle

sys.path.insert(0, '/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV')
import in_situ_analysis_PIVintegration
#import in_situ_analysis_PIVintegration as is3      # reading in with different name confuses pickling 

#reload(is3)

# =====================================================================================================

# Parameters: 

# Input Directory --------------------------------------------------------
# file struncture:
    # rootdir > list of profiles (profile) > shrink (subdir) > images (img) + ROIs > ROIs_classified

# Use if you want to make a list of shrink directoires within a parent directory
rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/video_data/sorted_videos/fps_20'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/test_folder'
#rootdir = None
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/test_folder'

# Use if you want to manually create a list of directories:
#dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_mini']
#dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test']
#dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200','/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200_400', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_400_600', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test']
#dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_400_600', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800']

# File Names --------------------------------------------------------
zoop_dat = 'zoop_30-5000.dat'

#classification = 'ROIs_classified/predictions.csv'         # copepod, blob
#classification = 'ROIs_classified6/predictions.csv'
#classification = 'ROIs_classified_25Jan2023/predictions.csv'
classification = 'ROIs_classified_6MAR2023/predictions.csv'

CTD = '/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts'

# Ouput Location --------------------------------------------------------
output_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output'
#output_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output_tests'

# -----------------------------
# this eventually needs to be changed to include the 5000 frame vids
max_frames = 10000
#max_frames = 2000       # skipping the really long videos for now 
# -----------------------------

# ==============================================================
start_time = time.time()

# Generate a list of shrink directories to process 
shrink_dirs_to_process = []
shrink_dirs_to_skip = []

if rootdir is not None:
    for profile in os.listdir(rootdir):
        #os.listdir(os.path.join(rootdir,profile))
        for subdir in os.listdir(os.path.join(rootdir,profile)):
            if subdir == 'shrink':
                if len(os.listdir(os.path.join(rootdir,profile,subdir))) < max_frames and len(os.listdir(os.path.join(rootdir,profile,subdir))) > 2:        # 2 bc the ROIs folder and .dat file are going to exist (fixed this downstream so eventually change back to zero)
                    shrink_dirs_to_process.append(os.path.join(rootdir,profile,subdir))
                else:
                    skipped_dirs = [str(os.path.join(rootdir,profile)), 'skipped']
                    shrink_dirs_to_skip.append(skipped_dirs)
elif dir_list is not None:
    shrink_dirs_to_process = dir_list

print("")
print("==============================================")
print("")
print("Profiles to Process: "+str(shrink_dirs_to_process))

# Create new output directory 
time_now = str(datetime.datetime.now())
output_path = os.path.join(output_dir, time_now)
os.mkdir(output_path)
print("Output Directory Created: " + str(output_path))
print("")

# Create .txt file with user input run information
text = input("Please enter batch analysis notes: ")
with open(os.path.join(output_path, "analysis_info.txt"), mode = "w") as f:
    f.write(text)

print("")

# ==============================================================

# Process shrink directories

analysis_objs_created = []
analysis_objs_failed = []
video_dic = {}

for shrink in shrink_dirs_to_process:
    
    shrink_dir = str(shrink+"/")        # eg: /home/dg/Wyeth2/IN_SITU_MOTION/test_folder/1501326258/shrink/
    print("==============================================")
    print("")
    print("ATTEMPTING TO PROCESS: "+shrink_dir)
    print("")
    print("==============================================")
    
    try:
        # 1) Create an analysis object for the video 
        print("Loading data....")
        video = in_situ_analysis_PIVintegration.Analysis(zoop_dat_file = os.path.join(shrink,zoop_dat), 
            snow_directory = shrink,
            class_file = os.path.join(shrink,classification),
            CTD_dir = CTD)   
        
        # 2) Call methods
        print("Matching classifications....")
        video.assign_class_and_size()
        print("Matching nearest CTD casts....")
        video.assign_chemistry()             
        print("Generating/removing background flow....")
        video.remove_flow()                         # this is a really slow step
        print("Converting to physical motion...")
        video.convert_to_physical()
        
        # 3) Store analysis and key information in lookup table
        print("Storing video information....")
        #video_dic[str(Path(shrink).parent)[-10:]] = video                         # this will be helpful if I want to pickle the whole run
        profile = str(Path(shrink).parent)[-10:]
        datetime = video.profile
        avg_frm_rate = video.avg_frame_rt
        total_frames = len(video.full_flowfield.tif_list)
        num_paths = len(video.zoop_paths)
        depth = video.depth_avg
        oxygen = video.oxygen_mgL_avg
        temp = video.temp_avg
        nearest_ctd = video.nearest_earlier_cast
        nearest_ctd_offset = video.time_offset
        
        analysis_info = [profile, datetime, avg_frm_rate, total_frames, num_paths, depth, oxygen, temp, nearest_ctd, nearest_ctd_offset]
        #analysis_info = [str(Path(shrink).parent)[-10:], video.profile, video.depth_avg, video.oxygen_mgL_avg, len(video.zoop_paths)]
        analysis_objs_created.append(analysis_info)
        
        # 4) Save pickle
        print("Pickling....")
        pickle_name = str(str(Path(shrink).parent)[-10:]+'.pickle')
        pickle_path = os.path.join(output_path,pickle_name)
        pickle_file = open(pickle_path, 'wb')
        pickle.dump(video, pickle_file)
        pickle_file.close()
        print(" *** SUCCESSFULLY PROCESSED/SAVED VIDEO: " + str(analysis_info))
        
    except:
            # Still store a list of videos that fucked up in some way -- maybe I can try to track where it errored out
            failed_dirs = [str(Path(shrink).parent), 'failed']
            #failed_dirs = [str(Path(shrink).parent)[-10:], 'failed']
            analysis_objs_failed.append(failed_dirs)
            print(" *** FAILED TO PROCESS VIDEO: " + shrink_dir)
        
    reload(in_situ_analysis_PIVintegration)        # having an issue running videos sequentially without clearning the environment?

# Save final summary tables
print("Saving Lookup Tables ... ")
processed_lookup = np.array(analysis_objs_created)
processed_lookup_file = 'processed_lookup_table.csv'
processed_lookup_path = os.path.join(output_path, processed_lookup_file)
np.savetxt(processed_lookup_path, processed_lookup, delimiter=',', fmt='%s', header='profile, datetime, avg_frm_rate, total_frames, num_paths, depth, oxygen, temp, nearest_ctd, nearest_ctd_offset')
#np.savetxt(processed_lookup_path, processed_lookup, delimiter=',', fmt='%s', header='Profile,Datettime,Depth,Oxygen,ZoopPaths')

skipped = np.array(shrink_dirs_to_skip)
failed = np.array(analysis_objs_failed)
skipped_lookup = np.empty((0,2))

if len(skipped) > 0:
    skipped_lookup = np.concatenate((skipped_lookup, skipped), axis=0)

if len(failed) > 0:
    skipped_lookup = np.concatenate((skipped_lookup, failed), axis=0)

skipped_lookup_file = 'skipped_lookup_table.csv'
skipped_lookup_path = os.path.join(output_path, skipped_lookup_file)
np.savetxt(skipped_lookup_path, skipped_lookup, delimiter=',', fmt='%s', header='Profile,ErrorType')
print("Script Done")

# ==============================================================

# Final readout

print("==============================================")
print("")
print("FINAL READOUTS:")
print("Attempted Profiles: " + str(shrink_dirs_to_process))
print("Skippped Profiles: " + str(shrink_dirs_to_skip))
print("Processed Profiles: " + str(np.array(analysis_objs_created)))
print("Failed Profiles: " + str(analysis_objs_failed))
print("Run time: %s seconds" % (time.time() - start_time))
print("")
print("==============================================")


# End of script 

# -----------------------------------------------------------------------------------------

# BUGBASHING:

# 3) (solved..... try not using .py) Warning at the end of the whole script running 
    # /usr/local/bin/python3.8: Error while finding module specification for 'batch_run_PIA_w_PIV.py' (ModuleNotFoundError: __path__ attribute 
    # not found on 'batch_run_PIA_w_PIV' while trying to find 'batch_run_PIA_w_PIV.py')

# (solved) 2) Warning when I run video.remove_flow() - still runs but need to figure out 
# https://stackoverflow.com/questions/31814837/numpy-mean-of-empty-slice-warning -- just ignored error
    # /home/dg/.local/lib/python3.8/site-packages/numpy/core/fromnumeric.py:3474: RuntimeWarning: Mean of empty slice.
    #   return _methods._mean(a, axis=axis, dtype=dtype,
    # /home/dg/.local/lib/python3.8/site-packages/numpy/core/_methods.py:189: RuntimeWarning: invalid value encountered in double_scalars
    #   ret = ret.dtype.type(ret / rcount)

# (solved) 1) this has been happening sometimes when I dont quit python. Something must not be cleared at the beginning of in_situ_analysis when it is rerun
# is there a command to clear env before looping through the script each time? would I even want that?
    # Traceback (most recent call last):
    #   File "<stdin>", line 22, in <module>
    #   File "/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV/in_situ_analysis_PIVintegration.py", line 329, in assign_classification
    #     rois = self.np_class_rows[(self.np_class_rows[:,-3]) == self.frame, :]   # save lines of np_class_rows at correct frame
    # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
# Fixed this by reloading is3 at the end of each loop -- I dont know if thats a bandaid or a legit way to solve this, regarless working now

