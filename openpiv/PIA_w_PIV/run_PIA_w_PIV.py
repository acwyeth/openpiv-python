
# Seperating out lines of code that run/test in_situ_analysis module

# ==================================================================================
# Notes from Anaylsis:

# Note for Amy:
# 1) cd dg/data2/dg/Wyeth2/GIT_repos_insitu/openpiv-python
# 2) python3.8
# 3) run code below

from importlib import reload
import sys

import matplotlib.pyplot as plt
import csv
import numpy as np
from statistics import mean, median
import pandas as pd
import math

# ----------

sys.path.insert(0, '/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV')
import in_situ_analysis_PIVintegration as is3

reload(is3)

# ------------

np.set_printoptions(suppress=True, linewidth=1000) 
np.set_printoptions(suppress=True, linewidth=75) 
pd.set_option('display.max_rows', 1000)

# ==================================================================================

# TESTS with everything run from scratch in 1537773747 folder

# NOTES: you have to update the start frame (line 323) in is3 for each video -- very annoying 
    # once I start working with full videos they will all start at the same frame number

# ------------
# motion_test 
    # ran! 
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.assign_classification()
test.assign_chemistry()             # turn off some of the output
test.remove_flow()                  # run time is fairly slow , get a RuntimeWarning

test.profile                        
test.nearest_earlier_cast           
test.oxygen_mgL_avg
test.depth_avg
test.zoop_paths[0].x_motion
test.zoop_paths[0].classification

# ------------
# motion_mini with missing frame
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_mini_missing/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_mini_missing',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_mini_missing/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.assign_classification()
test.assign_chemistry()             # turn off some of the output
test.remove_flow()    

# ------------
# shrink_100
    # ran!
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.assign_classification()        
# I get this error when I dont restart the ptyon session ? Must not be clearing something? 
    #  File "/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV/in_situ_analysis_PIVintegration.py", line 329, in assign_classification
    #    rois = self.np_class_rows[(self.np_class_rows[:,-3]) == frame, :]   # save lines of np_class_rows at correct frame
    #IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed 
test.assign_chemistry()
test.remove_flow()

test.profile                        
test.zoop_paths[0].x_motion
test.zoop_paths[0].classification

# ------------
#shrink_200
    # ran! 
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.assign_classification()
test.assign_chemistry()
test.remove_flow()

test.profile                        
test.zoop_paths[0].x_motion
test.zoop_paths[0].classification

# ------------
#shrink_200_400
    # 
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200_400/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200_400',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200_400/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.assign_classification()
test.assign_chemistry()
test.remove_flow()

test.profile                        
test.zoop_paths[0].x_motion
test.zoop_paths[0].classification

# ------------
# shinkr_400_600
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_400_600/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_400_600',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_400_600/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.assign_classification()
test.assign_chemistry()
test.remove_flow()

test.profile                        
test.zoop_paths[0].x_motion
test.zoop_paths[0].classification

# ------------
# shrink_600_800
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.assign_classification()
test.assign_chemistry()
test.remove_flow()

test.profile                        
test.zoop_paths[0].x_motion
test.zoop_paths[0].classification
