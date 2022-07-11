
# Script to batch process and store analysis objects for numerous videos 

# ACW 11 July 2022

# Notes:
    # Need to think about how I want to be able to access analysis objects for final stats/analysis 
    
# Execute:
    # 1) cd dg/data2/dg/Wyeth2/GIT_repos_insitu/openpiv-python
    # 2) python3.8

    
# =====================================================================================================

from importlib import reload
import sys
import os
from pathlib import Path

sys.path.insert(0, '/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV')
import in_situ_analysis_PIVintegration as is3

reload(is3)

# =====================================================================================================
# file struncture:
    # rootdir > list of profiles (profile) > shrink (subdir) > images (img)

# Use if you want to make a list of shrink directoires within a parent directory
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/test_folder'
rootdir = None

# User if you want to manually create a list of directories:
#dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100']
#dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test']
#dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200','/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200_400', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_400_600', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test']
dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_400_600', '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800']

# ----------------------------------------------------------------------------

# File names:
zoop_dat = 'zoop_30-5000.dat'
classification = 'ROIs_classified/predictions.csv'
CTD = '/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts'

# ==============================================================
#start_time = time.time()

# Generate a list of shrink directories to process 
shrink_dirs_to_process = []
shrink_dirs_to_skip = []

if rootdir is not None:
    for profile in os.listdir(rootdir):
        #os.listdir(os.path.join(rootdir,profile))
        for subdir in os.listdir(os.path.join(rootdir,profile)):
            if subdir == 'shrink':
                if len(os.listdir(os.path.join(rootdir,profile,subdir))) < max_frames:
                    shrink_dirs_to_process.append(os.path.join(rootdir,profile,subdir))
                else:
                    shrink_dirs_to_skip.append(os.path.join(rootdir,profile))
elif dir_list is not None:
    shrink_dirs_to_process = dir_list

print("Profiles to Process:"+str(shrink_dirs_to_process))

# -----------------------------------------------------------------------------------------

for shrink in shrink_dirs_to_process:
    shrink_dir = str(shrink+"/")
    print("PROCESSING:"+shrink_dir)
    
    # Store unique analysis name
    line = Path(shrink)
    line = str(line.parent)
    name = line[-10:]
    
    # 1) Create an unique analysis object for the video 
    name = is3.Analysis(zoop_dat_file = os.path.join(shrink,zoop_dat), 
        snow_directory = shrink,
        class_file = os.path.join(shrink,classification),
        CTD_dir = CTD)
    
    # name = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000.dat', 
    #     snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test',
    #     class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROIs_classified/predictions.csv',
    #     CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')
        
    # 2) Call methods
    name.assign_classification()
    name.assign_chemistry()             
    #name.remove_flow()

# *** Bug: this has been happening sometimes when I dont quit python. Something must not be cleared at the beginning of in_situ_analysis when it is rerun

# Traceback (most recent call last):
#   File "<stdin>", line 22, in <module>
#   File "/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV/in_situ_analysis_PIVintegration.py", line 329, in assign_classification
#     rois = self.np_class_rows[(self.np_class_rows[:,-3]) == self.frame, :]   # save lines of np_class_rows at correct frame
# IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed


# -----------------------------------------------------------------------------------------

print("Skippped Profiles:"+str(shrink_dirs_to_skip))
print("DONE")
#print("Run time: %s seconds" % (time.time() - start_time))