
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
# video breakinng in batch script

# Removing background flow....
# Broken frame pair: ['SHRINK-30-SPC-UW-1501426334918716-258237003382-000008.tif'], []
# Broken frame pair: [], ['SHRINK-30-SPC-UW-1501426334958056-258237053386-000010.tif']
# Broken frame pair: ['SHRINK-30-SPC-UW-1501426359955784-258262355489-000516.tif'], []
# Segmentation fault (core dumped)

# also closes my python session if Im running from within python 
# https://stackoverflow.com/questions/13654449/error-segmentation-fault-core-dumped 

# self.full_flowfield.get_flow(p.frames[l], p.x_pos[l], p.y_pos[l])
    # tree = spatial.KDTree(coordinates)
    
# I think the issue is that in this particular video the first array is empty, so when it pulls the x,y coords there is nothing and its explodes
# because the x,y grids are always the same I could hard code these in? 



test_zoops = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/test_folder/1501426321/shrink/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/test_folder/1501426321/shrink',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/test_folder/1501426321/shrink/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test_zoops.remove_flow()


test = is3.Flowfield_PIV_Full(directory='/home/dg/Wyeth2/IN_SITU_MOTION/test_folder/1501426321/shrink')

test.flowfield_full_np
test.u_flow_raw
test.u_flow_smooth[1]
u_point_flow = test.u_flow_smooth[1].reshape(20,)
v_point_flow = test.v_flow_smooth[1].reshape(20,)

test.flowfield_full_np[0,0,:]

coordinates = list(zip(self.flowfield_full_np[0,0,:], self.flowfield_full_np[0,1,:]))
        
        print(coordinates)
        print("check4")
        tree = spatial.KDTree(coordinates)
        print("check5")
        pos_ind = tree.query([(self.x_pos,self.y_pos)])[1][0]

# create a single PIV flowfield
self.full_flowfield = Flowfield_PIV_Full(self.snow_directory)

for p in self.zoop_paths:
    # store PIV flow at specific space/time localizations
    for l in range(len(p.frames)):
        self.full_flowfield.get_flow(p.frames[l], p.x_pos[l], p.y_pos[l])
        p.x_flow_smoothed[l] = self.full_flowfield.point_u_flow
        p.y_flow_smoothed[l] = self.full_flowfield.point_v_flow
    
    # calculate zooplankton motion (PTV of zoop paths minus PIV of snow particles)
    p.x_motion = (p.x_vel_smoothed - p.x_flow_smoothed)
    p.y_motion = (p.y_vel_smoothed - p.y_flow_smoothed)




# Flowfield test
test = is3.Flowfield_PIV_Full(directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_mini_missing')
test.smooth_flow()
test.get_flow(2, 100, 200)

test.flowfield_full_np
test.flowfield_full_np.shape
test.flowfield_full_np[:,2,:].reshape(15,4,5)
test.u_flow_raw
test.u_flow_smooth

# ------------
# motion_test 
    # ran! 
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.assign_classification()
test.assign_chemistry()             # turn off some of the output
test.remove_flow()                  # run time is fairly slow -- still using old Flowfield class -- need to update next
test.convert_to_physical()

test.zoop_paths[1].classification
test.zoop_paths[1].x_flow_raw
test.zoop_paths[1].x_flow_smoothed      # this will change a little 
test.zoop_paths[1].x_motion
test.zoop_paths[1].x_motion_phys

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
test.assign_chemistry()
test.remove_flow()
test.convert_to_physical()


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
