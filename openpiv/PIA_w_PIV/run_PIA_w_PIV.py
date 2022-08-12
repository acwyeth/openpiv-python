
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


np.set_printoptions(threshold=sys.maxsize)

# ==================================================================================

test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537773747/shrink/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537773747/shrink',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537773747/shrink/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.assign_classification()
test.assign_chemistry()             
test.remove_flow()                  
test.convert_to_physical()

# ---------------------------------------------------------------

# videos with NaN flowfields:
    # ['1536826760' '1537804398']
    # 6760 -- this one should break , 400 frames and very fragmented and broken/corrupted frames 
    # 4398 -- this one should work, maybe there is too big of a gap (472-487) (greater than the knt spacing - 10)
        # hmm not sure the best fix for this one -- dont want to increase my knt placement to span the gap

test = is3.Flowfield_PIV_Full(directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537804398/shrink')

from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
      
test.u_flow_raw = test.flowfield_full_np[:,2,:].reshape(test.vid_length,4,5)
test.v_flow_raw = test.flowfield_full_np[:,3,:].reshape(test.vid_length,4,5)
        
# empty array to store smoothed values
test.u_flow_smooth = np.empty_like(test.u_flow_raw)
test.v_flow_smooth = np.empty_like(test.v_flow_raw)

# TEMPORAL SMOOTHING
# this is a placeholder for the real frame number, which I don't think we need here, its just a pseudo x-axis
frames = list(range(test.u_flow_raw.shape[0]))

# repeat for each of the 20 (5x4) sptials grids
for i in range(test.u_flow_raw.shape[1]):
    for j in range(test.u_flow_raw.shape[2]):
        u_grid_thru_time = test.u_flow_raw[:,i,j]
        v_grid_thru_time = test.v_flow_raw[:,i,j]
        
        flow_knts = []
        flow_knt_smooth = 10
        flow_num_knts = int((frames[-1] - frames[0])/flow_knt_smooth)
        flow_knt_space = (frames[-1] - frames[0])/(flow_num_knts+1)
        for k in range(flow_num_knts):
            flow_knts.append(flow_knt_space*(k+1) + frames[0])
        
        # assign zero weight to nan values (https://gemfury.com/alkaline-ml/python:scipy/-/content/interpolate/fitpack2.py)
        wu = np.isnan(u_grid_thru_time)
        u_grid_thru_time[wu] = 0.
        wv = np.isnan(v_grid_thru_time)
        v_grid_thru_time[wv] = 0.
        
        # assign zero weight to all 0 values (the converted nans and piv zeros -- need to think about why those are here)
        wu = np.array([i==0 for i in u_grid_thru_time])
        wv = np.array([i==0 for i in v_grid_thru_time])
        
        # Same smoothing method as for zooplankton tracks
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
        u_flow_spline = LSQUnivariateSpline(frames, u_grid_thru_time, flow_knts, w=~wu, k=1)       # calculate spline for observed flow
        u_flow_output = u_flow_spline.__call__(frames)
        v_flow_spline = LSQUnivariateSpline(frames, v_grid_thru_time, flow_knts, w=~wv, k=1)
        v_flow_output = v_flow_spline.__call__(frames)
        
        test.u_flow_smooth[:,i,j] = u_flow_output
        test.v_flow_smooth[:,i,j] = v_flow_output

plt.plot(frames, test.u_flow_raw[:,1,1])
plt.plot(frames, test.u_flow_smooth[:,1,1])






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
