
# Seperating out lines of code that run/test in_situ_analysis module

# ==================================================================================
# Notes from Anaylsis:


# ADD A PARAMETER CALLED START FRAME 
    # If I can't get T3D to read in starting frame number then add a parameter I define in this script so I dont have to tweak and internal line of code in isa everytime I run it 

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

#import PIV_w_Zoop_Mask_for_PIA as piv
#reload(piv)

# ------------

np.set_printoptions(suppress=True, linewidth=1000) 
np.set_printoptions(suppress=True, linewidth=75) 
pd.set_option('display.max_rows', 1000)

# ==================================================================================

# 100 frame motion test 
test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROI_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.remove_flow()                  # run time is fairly slow 
test.assign_classification()
test.assign_chemistry()

test.profile                        
test.nearest_earlier_cast           
test.oxygen_mgL_avg
test.depth_avg

test.zoop_paths[0].x_motion
test.zoop_paths[0].classification

# ------------

test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/zoop_30-5000_joined_la2.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

test.remove_flow()                  # run time is fairly slow 
test.assign_classification()
test.assign_chemistry()

# ------------
# short clip with dropped frames 

# I dont know why but I am getting an error readying in .dat files when frames are dropped? 
# Its the spacing of the internal knots -- I was working on this a while ago and I THINK I needed to change the look ahead paramter in T3D from 5 to 2?
# I THINK THIS WORKED?
# I think I am getting to the point where I just need to rerun all my test clips -- too many little things have changes

test = is3.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100_rerun/zoop_30-5000_DG_la2.dat', 
    snow_directory='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100_rerun',
    class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100_rerun/ROIs_classified/predictions.csv',
    CTD_dir='/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts')

#ps = sio.loadmat('/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100_rerun/zoop_30-5000_DG_la2.dat',variable_names=['data'],simplify_cells=True)['data']
#from in_situ_analysis_PIVintegration import Path
#paths=[Path(i, ps[i], k_val=1, knot_smooth=3, verbose=True) for i in range(len(ps))]

test.remove_flow()                  # run time is fairly slow 
test.assign_classification()
test.assign_chemistry()

# ==================================================================================


# NOTE : remember to change the number of frames adding/subtracting (annoying)

# FRAMES 200-300 -- dropped frames but not patically corrupted frames
test = isa.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200/zoop_30-5000_DG.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200/snow_0-3_DG.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_200/ROIs_classified/predictions.csv')
test.assign_classification()

# same thing is happening -- frame offset after dropped frames
test.zoop_paths[6].frames # bridges dropped frames? Havent seen that before
test.zoop_paths[11].frames # after dropped -- none are matched 

# ------------------

# FIRST 100 frames (rerun)
test = isa.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100_rerun/zoop_30-5000_DG.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100_rerun/snow_0-3_DG.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100_rerun/ROIs_classified/predictions.csv')
test.assign_classification()

# ------------------

# 200 frames tests 
test = isa.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/zoop_30-5000_join-125-125.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/snow_0-3.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/ROIs_classified/predictions.csv')
test2 = is2.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/zoop_30-5000_joined_la2.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/snow_0-3.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_600_800/ROIs_classified/predictions.csv')
# if there is one zoop path where the internal knots dont work, it breaks the code 
    # currently the case -- not sure the best way around it if I want to keep the knot spacing as it
    # I think it's usually where there are dropped frames?

# ------------------
# 100 frame motion test
# WORKING
# 
# maybe RERUN!! The snow paths are a little wonky with huge gaps in frames, Im wondering if I tried to join paths at one point and saved that? 
# reran from .pst file on March 28

test = isa.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/snow_0-3.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROI_classified/predictions.csv')
test2 = is2.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/snow_0-3.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROI_classified/predictions.csv')
# look ahead 5 (for snow paths)
test2 = is2.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/snow_0-3_la5.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROI_classified/predictions.csv')
# look haead 2 (for snow)
test2 = is2.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/snow_0-3_la2.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROI_classified/predictions.csv')
test2 = is2.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/zoop_30-5000_joined.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/snow_0-3_la2.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/ROI_classified/predictions.csv')

# ------------------
# FIRST 100 frames
test = isa.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/zoop_30-5000.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/snow_0-3.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/ROIs_renamed_Classified/predictions.csv')
test2 = is2.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/zoop_30-5000.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/snow_0-3.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/ROIs_renamed_Classified/predictions.csv')

# with new frame naming in .fos-part file -- actually the same
# test = isa.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/zoop_30-5000_NEW_FRAME.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/snow_0-3_NEW_FRAME.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink_100/ROIs_renamed_Classified/predictions.csv')

# ------------------
# full shrink video
# also has some weird snow paths with huge gaps in the frames? Look into 
test = isa.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink/zoop_30-5000.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink/snow_0-3.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink/ROIs_classified/predictions.csv')
test2 = is2.Analysis(zoop_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink/zoop_30-5000_join_125_125.dat', snow_dat_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink/snow_0-3_la2.dat', class_file='/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink/ROIs_classified/predictions.csv')

# I think this is error is gone now! I remade files 
    # Trying to load snow_dat_file:  /home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/shrink/snow_0-3.dat
    # /home/dg/.local/lib/python3.6/site-packages/scipy/interpolate/fitpack2.py:253: UserWarning:
    # The maximal number of iterations maxit (set to 20 by the program)
    # allowed for finding a smoothing spline with fp=s has been reached: s
    # too small.
    # There is an approximation returned but the corresponding weighted sum
    # of squared residuals does not satisfy the condition abs(fp-s)/s < tol.
    # warnings.warn(message)

# I think the flowfield looks good except the smoothing spline is too sensititve to outlier points. Also Im not sure what the above error means?
    # There are some massive outliers that the spline goes through -- NEED TO FIX 
    # Everything is very slowed down with this number of paths
    # Not sure what to make of the check_flow plot - its definitely centered around (0,0) but its also a mess with a lot of points with large residuals
    # The zooplankton plots are also hard to interpret -- need to think about other visulization

    # This warning (sometimes):
        #QApplication: invalid style override 'gtk' passed, ignoring it.
        #Available styles: Windows, Fusion

test.assign_classification()
test2.assign_classification()

        # worked -- looks like 100% match
        # sometimes if I try to run again without restarting python I get this error:
            #Traceback (most recent call last):
            #File "<stdin>", line 1, in <module>
            #File "/media/dg/data2/dg/Wyeth2/GIT_in_situ_motion/in_situ_analysis.py", line 275, in assign_classification
            #rois = self.np_class_rows[(self.np_class_rows[:,-3]) == frame, :]   # save lines of np_class_rows at correct frame
            #IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed

test.spatial_parsing_9bin(start_frame=0, end_frame=101)
test.spatial_parsing_9bin(start_frame=101, end_frame=201)
test.spatial_parsing_9bin(start_frame=0, end_frame=1001)    #massive outlier thows things off
test.spatial_parsing_9bin(start_frame=0, end_frame=100) 
test.spatial_parsing_9bin(start_frame=600, end_frame=1000)

test.df
test.x_avgs

# ==================================================================================

# Run methods
test.assign_classification()
    # there were no instances where there were multiple ROIs matched which is good
    # about half the paths had no matches though
        # either the whole path was matched or the whole path was missed
        # meaning there was a zooplanton swimming path but not an ROI saved within range

test.check_flow()                   # plotting snow points - flowfield (ideally these residuals are small)

test.plot_flow_position()           # visualizing spread of points used to generate flowfield

test.plot_motion(plot_flow_motion=True, plot_flow_position=False)   # with the exception of 4ish zoops swimming opposite the x-flow, most of the swimming is in the y-direction (and the speeds are slower), I think this makes sense
                                                                    # I do think there might be too many zoop paths tho?
test.plot_motion(plot_flow_motion=False, plot_flow_position=True)

# ==================================================================================
# ==================================================================================

# multiple regression test -- turned into exploration of outliers 

test.check_flow() # this method calculates the 'motion' of the snow particles -- aka flowfield with temporal interpolation removed 
test2.check_flow()

# snow paths using new spline with assigned internal knots
spacial_reg_x = []
spacial_reg_y = []

for p in range(len(test2.snow_paths)):                          
    for l in range(len(test2.snow_paths[p].frames)):
        spacial_reg_x.append([p, test2.snow_paths[p].frames[l], test2.snow_paths[p].x_pos[l], test2.snow_paths[p].y_pos[l], test2.snow_paths[p].x_vel_smoothed[l], test2.snow_paths[p].x_motion[l]])
        spacial_reg_y.append([p, test2.snow_paths[p].frames[l], test2.snow_paths[p].x_pos[l], test2.snow_paths[p].y_pos[l], test2.snow_paths[p].y_vel_smoothed[l], test2.snow_paths[p].y_motion[l]])

reg_x = pd.DataFrame(spacial_reg_x, columns = ['path', 'frames', 'x_pos', 'y_pos', 'x_vel', 'x_motion'])
reg_y = pd.DataFrame(spacial_reg_y, columns = ['path', 'frames', 'x_pos', 'y_pos', 'y_vel', 'y_motion'])

# Outlier Analysis
reg_x.loc[reg_x['x_motion'] == reg_x['x_motion'].max()]
reg_x.loc[reg_x['x_motion'] > 50]
reg_x.loc[reg_x['x_vel'] == reg_x['x_vel'].max()]
reg_x.loc[reg_x['x_vel'] == reg_x['x_vel'].min()]
reg_x.loc[reg_x['x_vel'] > 20]

i = 0
test2.snow_paths[i].x_spline.get_knots()
test2.snow_paths[i].frames

for p in range(len(test2.snow_paths)):
    print(test2.snow_paths[p].frames)
# looking for frames with large gaps -- there are quite a few, going to try changing look ahead

i = 0
plt.scatter(test2.snow_paths[i].frames,test2.snow_paths[i].x_pos)
plt.xlabel('Frame#')
plt.ylabel('raw x position')
plt.show()

plt.scatter(test2.snow_paths[i].frames,test2.snow_paths[i].x_vel_smoothed)
plt.xlabel('Frame#')
plt.ylabel('x-velocity smoothed')
plt.show()

# Spatial Interpolation
plt.scatter(reg_x['x_pos'], reg_x['x_motion'], marker='.')
plt.xlabel('x-position', fontsize=14)
plt.ylabel('x snow motion', fontsize=14)
plt.show()

plt.scatter(reg_x['y_pos'], reg_x['x_motion'], marker='.')
plt.xlabel('y-motion', fontsize=14)
plt.ylabel('x snow motion', fontsize=14)
plt.show()

plt.scatter(reg_y['x_pos'], reg_y['y_motion'], marker='.')
plt.xlabel('x-position', fontsize=14)
plt.ylabel('y snow motion', fontsize=14)
plt.show()

plt.scatter(reg_y['y_pos'], reg_y['y_motion'], marker='.')
plt.xlabel('y-position', fontsize=14)
plt.ylabel('y snow motion', fontsize=14)
plt.show()

# Multiple Regression
# x motion----------------------------------
from sklearn import linear_model
import statsmodels.api as sm

X = reg_x[['x_pos','y_pos']] 
Y = reg_x['x_motion']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

# y motion----------------------------------

X = reg_y[['x_pos','y_pos']] 
Y = reg_y['y_motion']
 
# with sklearn
regr = linear_model.LinearRegression()
regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# with statsmodels
X = sm.add_constant(X) # adding a constant
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

# ==================================================================================
# ==================================================================================

# Visualizing changes in smoothing parameters and methods 
new_x_spline = UnivariateSpline(test.snow_paths[i].frames, test.snow_paths[i].x_pos, s=100, k=3, ext=3)             
new_x_pos_smoothed=new_x_spline.__call__(test.snow_paths[i].frames, ext=3)                                 
new_x_vel_smoothed=new_x_spline.__call__(test.snow_paths[i].frames, nu=1, ext=3)   

i=12

knts = []
knt_smooth = 3
num_knt = int((test.zoop_paths[i].frames[-1] - test.zoop_paths[i].frames[0])/knt_smooth)
knt_space = (test.zoop_paths[i].frames[-1] - test.zoop_paths[i].frames[0])/(num_knt+1)
for k in range(num_knt):
    knts.append(knt_space*(k+1) + test.zoop_paths[i].frames[0])
    #knts.append(test.zoop_paths[0].frames + (knt_space*(k+1)))

i = 12
plt.scatter(test.zoop_paths[i].frames,test.zoop_paths[i].x_pos)
plt.xlabel('Frame#')
plt.ylabel('raw x position')
plt.show()

plt.scatter(test.zoop_paths[i].frames,test.zoop_paths[i].x_vel_smoothed)
plt.xlabel('Frame#')
plt.ylabel('x-velocity smoothed')
plt.show()

new_x_spline = LSQUnivariateSpline(test.zoop_paths[i].frames, test.zoop_paths[i].x_pos, knts, k=1, ext=3)             
new_x_spline.get_knots()
new_x_pos_smoothed=new_x_spline.__call__(test2.snow_paths[i].frames)                                 
new_x_vel_smoothed=new_x_spline.__call__(test2.snow_paths[i].frames, nu=1)  

plt.scatter(test2.snow_paths[i].frames,test2.snow_paths[i].x_pos)
plt.show()
plt.scatter(test2.snow_paths[i].frames,new_x_pos_smoothed)
plt.show()
plt.scatter(test2.snow_paths[i].frames,new_x_vel_smoothed)
plt.show()


test.zoop_paths[2].x_spline.get_knots()
test2.zoop_paths[2].x_spline.get_knots()

# snow
i = 20

plt.scatter(test2.snow_paths[i].frames,test2.snow_paths[i].x_pos)
plt.xlabel('Frame#')
plt.ylabel('raw x position')
plt.show()

plt.scatter(test2.snow_paths[i].frames,test2.snow_paths[i].x_vel_smoothed)
plt.xlabel('Frame#')
plt.ylabel('x-velocity smoothed')
plt.show()

plt.scatter(test2.snow_paths[i].frames,test2.snow_paths[i].x_motion)
plt.xlabel('Frame#')
plt.ylabel('x motion')
plt.show()

# zooplankton
len(test2.zoop_paths)
# 2, 3, 7, 8 are blobs

i = 12

plt.scatter(test2.zoop_paths[i].frames,test2.zoop_paths[i].x_pos)
plt.xlabel('Frame#')
plt.ylabel('raw x position')
plt.show()

plt.scatter(test2.zoop_paths[i].frames,test2.zoop_paths[i].x_vel_smoothed)
plt.xlabel('Frame#')
plt.ylabel('x-velocity smoothed')
plt.show()

plt.scatter(test2.zoop_paths[i].frames,test2.zoop_paths[i].x_motion)
plt.xlabel('Frame#')
plt.ylabel('x motion')
plt.show()






# ==================================================================================
# ==================================================================================

# plot spatial variability -- trying to get a hold on impact of spatial interpolation 
test.snow_paths[0].frames
test.snow_paths[0].x_pos
test.snow_paths[0].x_motion             # I definitely need to re-convince myself I'm doing this last step correctly
test.snow_paths[0].x_vel_smoothed       # Raw velocity 

# function to get flowfield at specific frames
self.flowfield.get_flow(p.frames)                       # this is a function I wrote that basically just calls this:
test.flowfield.x_vel_temporal_spline.__call__(frame)    # self is flowfield here 

#flowfield values
test.flowfield.frame_x_vels             # averaged velocity in each frame -- used to generated smoothing spline 
len(test.flowfield.frame_x_vels)        # 88 - Number of non-empty frames. There were some dropped frames in this clip (64-75)
test.flowfield.all_frames               # frames with paths in them (non-dropped mostly)
test.flowfield.x_vel_temporal_eval      # output of smoothing spline at each frame
test.flowfield.point_x_flow             # smaller array? Dont really know what's going on here 

test.zoop_paths[0].x_snow_flow 

# Im trying to figure out parameters to use
# I want to compare ~20-30 frames of snow motion to the temporal flowfield at those frames
# and then plot the variability in differences in 2D space 

# Write a function that:
    # Inputs: quadrant boundary, # frames to pool 
    # Outputs: for each period of time (start with whole clip), define quadrants (need to be able to modify this), and pool raw speed date

# I have to modify the method when I want a different number of bins, but couldnt figure out how to build that in yet
# I think the function would just look inside one rectagle -- then I would call multiple boxes and combine the output -- just gonna stick with this for now
def spatial_parsing(bin_num=9, x_pixels=876, y_pixels=650, start_frame=1, end_frame=50):
    #bins = [i for i in range(bin_num)]
    x_bins = list(np.linspace(0,x_pixels,(int(math.sqrt(bin_num)+1))))
    y_bins = list(np.linspace(0,y_pixels,(int(math.sqrt(bin_num)+1))))
    print(x_bins)
    print(y_bins)
    #columns = ['bin#', 'frame', 'x_pos', 'y_pos', 'x_vel_smoothed', 'y_vel_smoothed']
    data = []
    for p in range(len(test.snow_paths)):                           # paths in video clip
        for l in range(len(test.snow_paths[p].frames)):             # localizations within each path
            if test.snow_paths[p].frames[l] > start_frame and test.snow_paths[p].frames[l] < end_frame:    
                if test.snow_paths[p].x_pos[l] > x_bins[0] and test.snow_paths[p].x_pos[l] < x_bins[1] and test.snow_paths[p].y_pos[l] > y_bins[0] and test.snow_paths[p].y_pos[l] < y_bins[1]:
                    #print('Bin 1')
                    data.append([1, test.snow_paths[p].frames[l], test.snow_paths[p].x_pos[l], test.snow_paths[p].y_pos[l], test.snow_paths[p].x_vel_smoothed[l], test.snow_paths[p].y_vel_smoothed[l]])
                elif test.snow_paths[p].x_pos[l] > x_bins[1] and test.snow_paths[p].x_pos[l] < x_bins[2] and test.snow_paths[p].y_pos[l] > y_bins[0] and test.snow_paths[p].y_pos[l] < y_bins[1]:
                    #print('Bin 2')
                    data.append([2,test.snow_paths[p].frames[l], test.snow_paths[p].x_pos[l], test.snow_paths[p].y_pos[l], test.snow_paths[p].x_vel_smoothed[l], test.snow_paths[p].y_vel_smoothed[l]])
                elif test.snow_paths[p].x_pos[l] > x_bins[2] and test.snow_paths[p].x_pos[l] < x_bins[3] and test.snow_paths[p].y_pos[l] > y_bins[0] and test.snow_paths[p].y_pos[l] < y_bins[1]:
                    #print('Bin 3')
                    data.append([3, test.snow_paths[p].frames[l], test.snow_paths[p].x_pos[l], test.snow_paths[p].y_pos[l], test.snow_paths[p].x_vel_smoothed[l], test.snow_paths[p].y_vel_smoothed[l]])
                elif test.snow_paths[p].x_pos[l] > x_bins[0] and test.snow_paths[p].x_pos[l] < x_bins[1] and test.snow_paths[p].y_pos[l] > y_bins[1] and test.snow_paths[p].y_pos[l] < y_bins[2]:
                    #print('Bin 4')
                    data.append([4,test.snow_paths[p].frames[l], test.snow_paths[p].x_pos[l], test.snow_paths[p].y_pos[l], test.snow_paths[p].x_vel_smoothed[l], test.snow_paths[p].y_vel_smoothed[l]])
                elif test.snow_paths[p].x_pos[l] > x_bins[1] and test.snow_paths[p].x_pos[l] < x_bins[2] and test.snow_paths[p].y_pos[l] > y_bins[1] and test.snow_paths[p].y_pos[l] < y_bins[2]:
                    #print('Bin 5')
                    data.append([5, test.snow_paths[p].frames[l], test.snow_paths[p].x_pos[l], test.snow_paths[p].y_pos[l], test.snow_paths[p].x_vel_smoothed[l], test.snow_paths[p].y_vel_smoothed[l]])
                elif test.snow_paths[p].x_pos[l] > x_bins[2] and test.snow_paths[p].x_pos[l] < x_bins[3] and test.snow_paths[p].y_pos[l] > y_bins[1] and test.snow_paths[p].y_pos[l] < y_bins[2]:
                    #print('Bin 6')
                    data.append([6,test.snow_paths[p].frames[l], test.snow_paths[p].x_pos[l], test.snow_paths[p].y_pos[l], test.snow_paths[p].x_vel_smoothed[l], test.snow_paths[p].y_vel_smoothed[l]])
                elif test.snow_paths[p].x_pos[l] > x_bins[0] and test.snow_paths[p].x_pos[l] < x_bins[1] and test.snow_paths[p].y_pos[l] > y_bins[2] and test.snow_paths[p].y_pos[l] < y_bins[3]:
                    #print('Bin 7')
                    data.append([7,test.snow_paths[p].frames[l], test.snow_paths[p].x_pos[l], test.snow_paths[p].y_pos[l], test.snow_paths[p].x_vel_smoothed[l], test.snow_paths[p].y_vel_smoothed[l]])
                elif test.snow_paths[p].x_pos[l] > x_bins[1] and test.snow_paths[p].x_pos[l] < x_bins[2] and test.snow_paths[p].y_pos[l] > y_bins[2] and test.snow_paths[p].y_pos[l] < y_bins[3]:
                    #print('Bin 8')
                    data.append([8,test.snow_paths[p].frames[l], test.snow_paths[p].x_pos[l], test.snow_paths[p].y_pos[l], test.snow_paths[p].x_vel_smoothed[l], test.snow_paths[p].y_vel_smoothed[l]])
                elif test.snow_paths[p].x_pos[l] > x_bins[2] and test.snow_paths[p].x_pos[l] < x_bins[3] and test.snow_paths[p].y_pos[l] > y_bins[2] and test.snow_paths[p].y_pos[l] < y_bins[3]:
                    #print('Bin 9')
                    data.append([9, test.snow_paths[p].frames[l], test.snow_paths[p].x_pos[l], test.snow_paths[p].y_pos[l], test.snow_paths[p].x_vel_smoothed[l], test.snow_paths[p].y_vel_smoothed[l]])
    return data

parsed = spatial_parsing()
df = pd.DataFrame(parsed, columns = ['bin#', 'frame', 'x_pos', 'y_pos', 'x_vel_smoothed', 'y_vel_smoothed'])
#df.to_csv(self.file_output)


# bar plot
avgs = df.groupby('bin#', as_index=False)['x_vel_smoothed','y_vel_smoothed'].mean()
x_avgs = df.groupby('bin#').x_vel_smoothed.agg(['count', 'mean', 'std']).reset_index()
y_avgs = df.groupby('bin#').y_vel_smoothed.agg(['count', 'mean', 'std']).reset_index()

plt.bar(x_avgs['bin#'], x_avgs['mean'], yerr=x_avgs['std'], align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xlabel('grid number')
plt.ylabel('mean x-velocity w/ SD')
plt.title('Start Frame: '+str(df.frame.min())+' - End Frame: '+str(df.frame.max()))
plt.show()

plt.bar(y_avgs['bin#'], y_avgs['mean'], yerr=y_avgs['std'], align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xlabel('grid number')
plt.ylabel('mean y-velocity w/ SD')
plt.show()

# plot together
fig, (ax1, ax2) = plt.subplots(2)
ax1.bar(x_avgs['bin#'], x_avgs['mean'], yerr=x_avgs['std'], align='center', alpha=0.5, ecolor='black', capsize=10)
#ax1.set_xlabel('grid number')
ax1.set_ylabel('mean x-velocity w/ SD')
ax1.set_title('Start Frame: '+str(df.frame.min())+' - End Frame: '+str(df.frame.max()))
ax2.bar(y_avgs['bin#'], y_avgs['mean'], yerr=y_avgs['std'], align='center', alpha=0.5, ecolor='black', capsize=10)
ax2.set_xlabel('grid number')
ax2.set_ylabel('mean y-velocity w/ SD')
plt.show()

#--------
# Bin by frames
bins = list(np.linspace(1,100,6))
cuts = pd.cut(df['frame'], bins=bins)
bin_frame_x_avg = df.groupby([cuts,'bin#']).x_vel_smoothed.agg(['count', 'mean', 'std']).reset_index()
bin_frame_y_avg = df.groupby([cuts,'bin#']).y_vel_smoothed.agg(['count', 'mean', 'std']).reset_index()

# grouped bar plot (by frame group and spatial bin)
# set width of bars
barWidth = 0.15
 
# set heights of bars
bars1 = bin_frame_x_avg.loc[0:7,'mean']
bars2 = bin_frame_x_avg.loc[8:15,'mean']
bars3 = bin_frame_x_avg.loc[16:23,'mean']
bars4 = bin_frame_x_avg.loc[24:31,'mean']
bars5 = bin_frame_x_avg.loc[32:39,'mean']
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

# Make the plot
plt.bar(r1, bars1, color='red', width=barWidth, edgecolor='white', label='1-20')
plt.bar(r2, bars2, color='orange', width=barWidth, edgecolor='white', label='20-40')
plt.bar(r3, bars3, color='yellow', width=barWidth, edgecolor='white', label='40-60')
plt.bar(r4, bars4, color='green', width=barWidth, edgecolor='white', label='60-80')
plt.bar(r5, bars5, color='blue', width=barWidth, edgecolor='white', label='80-100')
 
# Add xticks on the middle of the group bars
plt.xlabel('spatial bin', fontweight='bold')
plt.ylabel('average snow speed', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], list(bin_frame_x_avg['bin#'].unique()))
 
# Create legend & Show graphic
plt.legend()
plt.show()


# Try to think about more customizable




# -------------------------
#cast_dates = list(map(lambda val: val.datestring, self.all_ctd_data))
list(map(lambda val: val.x_pos, test.snow_paths))
x_position = test.snow_paths[0].x_pos
y_position = test.snow_paths[0].y_pos
x_vel_diff = test.snow_paths[0].x_vel_smoothed - test.flowfield.x_vel_temporal_spline.__call__(test.snow_paths[0].frames)
y_vel_diff = test.snow_paths[0].y_vel_smoothed - test.flowfield.y_vel_temporal_spline.__call__(test.snow_paths[0].frames)

x_position = test.snow_paths[10].x_pos
y_position = test.snow_paths[10].y_pos
x_vel_diff = test.snow_paths[10].x_vel_smoothed - test.flowfield.x_vel_temporal_spline.__call__(test.snow_paths[10].frames)
y_vel_diff = test.snow_paths[10].y_vel_smoothed - test.flowfield.y_vel_temporal_spline.__call__(test.snow_paths[10].frames)

plt.plot(x_position, x_vel_diff)
plt.scatter(x_position, x_vel_diff)
plt.xlabel('x-position')
plt.ylabel('x-velocity difference')
plt.show()

plt.plot(y_position, y_vel_diff)
plt.scatter(y_position, y_vel_diff)
plt.xlabel('y-position')
plt.ylabel('y-velocity difference')
plt.show()


# This is not what im trying to do... 
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')
ax1.set_xlabel('x-position')
ax1.set_ylabel('y-position', color='g')
ax2.set_ylabel('snow motion', color='b')
plt.show()

for i in range(len(test.snow_paths)):
    if 10 in test.snow_paths[i].frames:
        print(i)













# FRAME OFFSET THOUGHTS 

# Motion_test notes - no dropped frames in this clip
    # recreated .fos-part file and ROI images
    # now getting 100% matching!

# [0] - all copepod -               avg motion: -9, -2
# [1] - all copepod -               avg motion: 2, -1
# [2] - all blobs - stationary -    avg motion: -11, 0 (same as flow at that frame)
# [3] - all blobs - stationary -    avg motion: -11, 0
# [4] - copepod with 1 blob -       avg motion: -218, -3
# [5] - copepod with 2 blob -       avg motion: -6, 0
# [6] - copepod with 2 blob -       avg motion: 5, -1
# [7] - all copepod -               avg motion: 5, 0
# [8] - all copepod -               avg motion: 0, 5

# these results make sense to me 
    # there are two stationary blobs that are present in every frame and dont move (2&3)
    # there are a 2-3 copepod swimming but the paths are sometimes split
    # handful of incorrectly identified copepods that we see in the copepod paths (I counted ~5 copepods in the blobs folder)
    # which means my issues in the other clip have to do with dropped frames (which also makes sense)

test.zoop_paths[1].frames
test.zoop_paths[1].x_pos
test.zoop_paths[1].y_pos

# ---------------------------------------------------

# shrink_100 notes - there are dropped frames in this clip
    # zoop_paths[0] - all blobs
    # zoop_paths[1] - mix blobs/copepods
    # zoop_paths[2] - all copepods
    # zoop_paths[3] - not matched   - these paths are all after the frame drop
    # zoop_paths[4] - all blobs     - I think this path was OK because it was a stationary blob, so whatever frame offset didnt affect it 
    # zoop_paths[5] - not matched
    # zoop_paths[6] - not matched

test.zoop_paths[6].classification

# 0 and 4 is the same stuck blob

# paths 3-6 all skip frames in their paths
    # this seems like a seperate T3D issues? the skipped frames aren't corrupted 

# dropped/corrupted frames:
    # 63 - sometimes looks fine in folder, opens, but broken - Image data of dtype object cannot be converted to float
    # 64 - sometimes looks fine in folder, opens, but broken - Image data of dtype object cannot be converted to float
    # 65 - sometimtes looks fine in folder, opens, but broken - Image data of dtype object cannot be converted to float
    # 66-69 - dropped, wont open in folder, same error as above frames
    # 70-71 - sometimes looks fine in folder
    # 72-74 - dropped 

# frames 63-74 can't be opened by python (some are visible in the folder others are definitely dropped)
    # aka frames 64-75 in zoop_paths b/c of the one frame offset

# test.zoop_paths[3].frames starts at 79 -- should it start at 76? (3 frame offset)
    # Not sure where this bug is being introduced?
    # But I can fix it if I scale the frames to match this offset 
        # need to adjust for both corrupted frames (that should be skipped) and skipped frames (that shouldnt be)

#for p in test.zoop_paths:
p = test.zoop_paths[3]
for l in range(len(p.frames)):
    # Save frame, x, and y position of that localization
    #frame = (p.frames[l]-1)                                
    #frame = (p.frames[l]-4)                                # TRYING TO FIX OFFSET! - matches first 6 points    
    frame = (p.frames[l]-5)                                # TRYING TO FIX OFFSET! - matches last 3 points (there is a skipped frame seperating these)
    #print(frame)
    x_pos = p.x_pos[l]
    #print(x_pos)
    y_pos = p.y_pos[l]
    #print(y_pos)
    # Pull ROI infomration from frame number
    rois = test.np_class_rows[(test.np_class_rows[:,-3]) == frame, :]   # save lines of np_class_rows at correct frame
    #print(rois)
    roi = rois[(rois[:,-2] < (x_pos+2)) & (rois[:,-2] > (x_pos-2)) & (rois[:,-1] < (y_pos+2)) & (rois[:,-1] > (y_pos-2)),:]         # if the center of the ROI is within sq pixels of the localization -- match it
    #print(roi)
    if len(roi) == 1:
        p.classification[l] = roi[:,4][0]
        print('SUCCESS: Match found')
    if len(roi) == 0:
        print('ERROR: No match found')
    if len(roi) > 1:
        print('ERROR: more than more 1 ROI found')

p = test.zoop_paths[5]
for l in range(len(p.frames)):
    # Save frame, x, and y position of that localization
    #frame = (p.frames[l]-1)                                
    #frame = (p.frames[l]-5)                                # TRYING TO FIX OFFSET! - matches first 9
    #frame = (p.frames[l]-6)                                # TRYING TO FIX OFFSET! - matches 7th (after skipped frame)
    frame = (p.frames[l]-7)                                # TRYING TO FIX OFFSET! - matches last 2 (after skipped frame)
    #print(frame)
    x_pos = p.x_pos[l]
    #print(x_pos)
    y_pos = p.y_pos[l]
    #print(y_pos)
    # Pull ROI infomration from frame number
    rois = test.np_class_rows[(test.np_class_rows[:,-3]) == frame, :]   # save lines of np_class_rows at correct frame
    #print(rois)
    roi = rois[(rois[:,-2] < (x_pos+2)) & (rois[:,-2] > (x_pos-2)) & (rois[:,-1] < (y_pos+2)) & (rois[:,-1] > (y_pos-2)),:]         # if the center of the ROI is within sq pixels of the localization -- match it
    #print(roi)
    if len(roi) == 1:
        p.classification[l] = roi[:,4][0]
        print('SUCCESS: Match found')
    if len(roi) == 0:
        print('ERROR: No match found')
    if len(roi) > 1:
        print('ERROR: more than more 1 ROI found')

p = test.zoop_paths[6]
for l in range(len(p.frames)):
    # Save frame, x, and y position of that localization
    #frame = (p.frames[l]-1)                                
    #frame = (p.frames[l]-4)                                # TRYING TO FIX OFFSET! - matches first 3
    #frame = (p.frames[l]-5)                                # TRYING TO FIX OFFSET! - matches 4th
    frame = (p.frames[l]-6)                                # TRYING TO FIX OFFSET! - matches last 7
    #print(frame)
    x_pos = p.x_pos[l]
    #print(x_pos)
    y_pos = p.y_pos[l]
    #print(y_pos)
    # Pull ROI infomration from frame number
    rois = test.np_class_rows[(test.np_class_rows[:,-3]) == frame, :]   # save lines of np_class_rows at correct frame
    #print(rois)
    roi = rois[(rois[:,-2] < (x_pos+2)) & (rois[:,-2] > (x_pos-2)) & (rois[:,-1] < (y_pos+2)) & (rois[:,-1] > (y_pos-2)),:]         # if the center of the ROI is within sq pixels of the localization -- match it
    #print(roi)
    if len(roi) == 1:
        p.classification[l] = roi[:,4][0]
        print('SUCCESS: Match found')
    if len(roi) == 0:
        print('ERROR: No match found')
    if len(roi) > 1:
        print('ERROR: more than more 1 ROI found')


# ==================================================================================

test.zoop_paths[6].x_vel_smoothed
test.zoop_paths[6].x_snow_flow
test.zoop_paths[6].x_motion

test.zoop_paths[6].y_vel_smoothed
test.zoop_paths[6].y_snow_flow
test.zoop_paths[6].y_motion

test.snow_paths[2].x_vel_smoothed
test.snow_paths[2].x_motion


test.snow_paths
test.zoop_paths
len(test.zoop_paths)        # there are 24 zoop paths, this seems too high, either they are picking up smaller zoops or the 4 copepods I see are getting broken into numerous paths, also there is usually 3-4 zoop ROIs in a frame 

test.zoop_paths[0].x_pos.size
test.zoop_paths[0].x_vel_smoothed
test.zoop_paths[0].frames

test.flowfield.all_frames
test.flowfield.get_flow(test.zoop_paths[0].frames)
test.flowfield.point_flow[1]

