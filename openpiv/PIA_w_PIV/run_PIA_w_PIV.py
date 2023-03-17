
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
np.set_printoptions(suppress=True, linewidth=50) 
pd.set_option('display.max_rows', 1000)


np.set_printoptions(threshold=sys.maxsize)

# ==================================================================================

test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537804398/shrink/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537804398/shrink',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537804398/shrink/ROIs_classified2/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')
    
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')
    
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/test_folder/1537855340/shrink/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/test_folder/1537855340/shrink',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/test_folder/1537855340/shrink/ROIs_classified_6MAR2023/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/video_data/sorted_videos/fps_20/1537773747/shrink/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/video_data/sorted_videos/fps_20/1537773747/shrink',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/video_data/sorted_videos/fps_20/1537773747/shrink/ROIs_classified_6MAR2023/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

#test.assign_classification()
test.assign_class_and_size()
test.assign_chemistry()
test.remove_flow()           
test.convert_to_physical()

# ---------------------------------------------------------------

test.zoop_paths[33].classification
test.zoop_paths[33].x_vel_smoothed
test.zoop_paths[33].x_flow_smoothed

test.zoop_paths[1].u_flow_raw
test.zoop_paths[1].u_flow_smoothed      # this will change a little 
test.zoop_paths[1].x_motion
test.zoop_paths[1].x_motion_phys

test.zoop_paths[33]

# SMOOTH FINAL SPEED VECTOR ?? 
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline

path_data = test.zoop_paths[33]
knts = []
knt_smooth = 3
num_knt = int((path_data.frames[-1] - path_data.frames[0])/knt_smooth)
knt_space = (path_data.frames[-1] - path_data.frames[0])/(num_knt+1)
for k in range(num_knt):
    knts.append(knt_space*(k+1) + path_data.frames[0])
test_spline = LSQUnivariateSpline(path_data.frames, path_data.speed, knts, k=1)


test_smooth_output = test_spline.__call__(path_data.frames)
test.zoop_paths[33].speed

plt.plot(path_data.frames, path_data.speed)
plt.plot(path_data.frames, test_smooth_output)

plt.plot(test.zoop_paths[1].frames, test.zoop_paths[1].speed_raw)
plt.plot(test.zoop_paths[1].frames, test.zoop_paths[1].speed)



# ---------------------------------------------------------------


# buidl assing_size metrics

for c in test.class_rows:
    # Pull filename 
    line = c[1]
    
    # Save frame number
    frame_tag = '_grp'                                          # frame number is listed directly before group number --  I think this is the easiest way to find it
    frame_tag_ind = line.find(frame_tag)
    frame_len= 6                                                # frame number is 6 digits long  
    frame_num = line[(frame_tag_ind-frame_len):frame_tag_ind]
    
    # Save major and minor semi-axis
    line = test.class_rows[5][1]
    ell_start_tag = '_e'
    ell_end_tag = '.tif'
    ell = line[line.find(ell_start_tag):line.find(ell_end_tag)]
    ell = ell[2:]
    ell = ell.replace("_", " ")
    ell_int = [float(word) for word in ell.split()]         # center y, center x, semi minor, semi major, angle 
    ell_length = ell_int[3] * 2
    ell_area = ell_int[3] * ell_int[2] * math.pi
    
    # Save center point 
    bbox_start_tag = '_r'     
    bbox_end_tag = 'e'
    bbox = line[line.find(bbox_start_tag):line.find(bbox_end_tag)]
    bbox = bbox[2:-1]                                           # remove _r and _ from beginning and end of string    
    bbox = bbox.replace("_", " ")
    bbox_int = [int(word) for word in bbox.split() if word.isdigit()]            
    # Explicitly define coordinate here (i and j are switched and coordinate system starts at top left corner)
        # This is unneccarily long but hopefully will make things clearer down the line
    x_beg = bbox_int[1]
    y_beg = bbox_int[0]
    height = bbox_int[2]
    width = bbox_int[3]
    
    # Center of ROI
    roi_cnt = [(x_beg + (width/2)), (y_beg + (height/2))]
    
    # Add columns to np_class_rows
    c.append(int(frame_num))
    c.append(roi_cnt[0])
    c.append(roi_cnt[1])
    c.append(ell_length)
    c.append(ell_area)
# Convert to numpy array
self.np_class_rows = np.array(self.class_rows, dtype=object)


# ---------------------------------------------------------------

# videos with NaN flowfields:
    # ['1536826760' '1537804398']
    # 6760 -- this one should break , 400 frames and very fragmented and broken/corrupted frames 
    # 4398 -- this one should work, maybe there is too big of a gap (472-487) (greater than the knt spacing - 10)
        # hmm not sure the best fix for this one -- dont want to increase my knt placement to span the gap

test = is3.Flowfield_PIV_Full(directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537804398/shrink_mini')

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
        
        # move knts if there is a big gap:
        for k in range(len(flow_knts)-1):
            if wu[int(flow_knts[k])] == True & wu[int(flow_knts[k+1])] == True:
                print('two in a row')
        
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

test.tif_list
test.flowfield_full_np[100:200]


is3.piv.PIV(frame1='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537804398/shrink_mini/SHRINK-8-SPC-UW-1537804406129508-810295774-000000.tif', frame2='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1537804398/shrink_mini/SHRINK-8-SPC-UW-1537804406176461-810345778-000001.tif', save_setting=False, display_setting=True, verbosity_setting=True)




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
