
# ACW 2023

# Script to read in directory of videos, calculate the PIV flowfield (and some other stats), and create .csv file with a flowfield summary

# ==========================================================

# import packages -----------------------------------
    # might not need all of these 

import scipy.io as sio
from statistics import mean, median
import numpy as np
from scipy import interpolate, spatial
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
import matplotlib.pyplot as plt
import csv
import math
import pandas as pd
import os
import sys
import datetime
import warnings

# import scripts ------------------------------------

sys.path.insert(0, '/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV')
from openpiv import tools, pyprocess, scaling, validation, filters
import PIV_w_Zoop_Mask_for_PIA as piv
#import CTD_matching_for_PIA as ctd

# set default parameters ----------------------------

warnings.simplefilter("ignore")
# numpy defaults
np.set_printoptions(suppress=True, linewidth=100)

#s_spline_flow=128       # flowfield smoothing parameter -- not using this class right now
#s_spline_flow=200

#verbose=False

max_frames = 1000

# PIV bins (5x4)
x_bins = np.array([182., 310., 438., 566., 694., 182., 310., 438., 566., 694., 182., 310., 438., 566., 694., 182., 310., 438., 566., 694.])
y_bins = np.array([517., 517., 517., 517., 517., 389., 389., 389., 389., 389., 261., 261., 261., 261., 261., 133., 133., 133., 133., 133.])

# ==========================================================

class Flowfield_PIV_Full():
    def __init__(self, directory=None):
        """Class to create a flowfield object for an entire video using PIV anaylsis. Temportal smoothing happens here
        """
        # Directory of tif images
        self.directory = directory
        
        # make an empty array to store full flowfeild information
        self.flowfield_full = []
        
        # create a list of all the frames in the video
        self.tif_list=[f for f in os.listdir(self.directory) if  \
                    (os.path.isfile(os.path.join(self.directory, f)) and f.endswith('.tif'))]
        self.tif_list.sort()
        first_roi_frame = self.tif_list[0][-10:-4]
        last_roi_frame = self.tif_list[-1][-10:-4]
        
        # STILL WRAPPING MY HEAD AROUND THIS 
        # calculate the number of .tif images there would be in frames were not missing (an inclusive difference)
        # because we're smoothing over missing frames for the flowfield, so the array needs to be shaped as if they are not missing
        self.vid_length = int(last_roi_frame) - int(first_roi_frame) + 1
        
        # for each frame pair calculate and store the flowfield 
        #for f in range(len(self.tif_list)-1):                      # this broke when frames were missing 
        for f in range(self.vid_length-1):
            
            # create an empty array 
            self.flow_layer = np.empty((5,20))      # 5 columns (x,y,u,v,mask) for 20 rows (5x4 flattened grid)
            
            roi_frame_a = int(first_roi_frame) + f 
            roi_frame_b = int(first_roi_frame) + f + 1
            
            roi_image_a = [f for f in os.listdir(self.directory) if  \
                        (os.path.isfile(os.path.join(self.directory, f)) and f.endswith(str("%06d"%(roi_frame_a))+'.tif'))]
            roi_image_b = [f for f in os.listdir(self.directory) if  \
                        (os.path.isfile(os.path.join(self.directory, f)) and f.endswith(str("%06d"%(roi_frame_b))+'.tif'))]
            
            # A check that both images exist
            image_a_len = len(roi_image_a)
            image_b_len = len(roi_image_b)
            
            if ((image_a_len>0) and (image_b_len>0)):                               # handles missing frames
                try:                                                                # handles corrupted frames 
                    self.frame_a = os.path.join(self.directory, roi_image_a[0])
                    self.frame_b = os.path.join(self.directory, roi_image_b[0])
                    
                    # Imported script that masks zooplankton ROIs from images and computes PIV flow analysis between consecutive frames (if both exist)
                    self.frame_flow = piv.PIV(frame1=self.frame_a, frame2=self.frame_b, save_setting=False, display_setting=False, verbosity_setting=False)
                    self.flow_layer = self.frame_flow.output     # this is a 20 (5x4 grid) x 5 (x, y, u, v, mask) array
                
                except:
                    print("Broken frame pair: "+str(roi_image_a)+", "+str(roi_image_b))
                    self.flow_layer.fill(np.NaN)
            
            else:
                print("Broken frame pair: "+str(roi_image_a)+", "+str(roi_image_b))
                self.flow_layer.fill(np.NaN)
            
            self.flowfield_full.append(self.flow_layer)
            
        # add one layer of NaNs for the last frame (versus treating it as broken frame pair)
        self.flowfield_full.append(np.full((5,20),np.nan))
        
        # convert to a 3D numpy array
        self.flowfield_full_np = np.array(self.flowfield_full)
        
        # call smoothing method
        self.smooth_flow()
    
    def smooth_flow(self):
        # raw PIV outputs
        #self.u_flow_raw = self.flowfield_full_np[:,2,:].reshape(len(self.tif_list),4,5)
        #self.v_flow_raw = self.flowfield_full_np[:,3,:].reshape(len(self.tif_list),4,5)
        self.u_flow_raw = self.flowfield_full_np[:,2,:].reshape(self.vid_length,4,5)
        self.v_flow_raw = self.flowfield_full_np[:,3,:].reshape(self.vid_length,4,5)
        
        # empty array to store smoothed values
        self.u_flow_smooth = np.empty_like(self.u_flow_raw)
        self.v_flow_smooth = np.empty_like(self.v_flow_raw)
        
        # TEMPORAL SMOOTHING
        # this is a placeholder for the real frame number, which I don't think we need here, its just a pseudo x-axis
        frames = list(range(self.u_flow_raw.shape[0]))
        
        # repeat for each of the 20 (5x4) sptials grids
        for i in range(self.u_flow_raw.shape[1]):
            for j in range(self.u_flow_raw.shape[2]):
                u_grid_thru_time = self.u_flow_raw[:,i,j]
                v_grid_thru_time = self.v_flow_raw[:,i,j]
                
                flow_knts = []
                #flow_knt_smooth = 3
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
                
                # NEW: assign zero weight to all 0 values (the converted nans and other zeros)
                wu = np.array([i==0 for i in u_grid_thru_time])
                wv = np.array([i==0 for i in v_grid_thru_time])
                
                # Same smoothing method as for zooplankton tracks
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
                u_flow_spline = LSQUnivariateSpline(frames, u_grid_thru_time, flow_knts, w=~wu, k=1)       # calculate spline for observed flow
                u_flow_output = u_flow_spline.__call__(frames)
                v_flow_spline = LSQUnivariateSpline(frames, v_grid_thru_time, flow_knts, w=~wv, k=1)
                v_flow_output = v_flow_spline.__call__(frames)
                
                self.u_flow_smooth[:,i,j] = u_flow_output
                self.v_flow_smooth[:,i,j] = v_flow_output

class Analysis():
    def __init__(self, rootdir=None):
        
        self.rootdir = rootdir
        self.flow_summary = []
        
        # for each video calculate flow stats/summary
        if rootdir is not None:
            for profile in os.listdir(rootdir):
            #for profile in filter(os.path.isdir, os.listdir(rootdir)):
                for subdir in os.listdir(os.path.join(rootdir,profile)):
                    if subdir == 'shrink':
                        if len(os.listdir(os.path.join(rootdir,profile,subdir))) < max_frames and len(os.listdir(os.path.join(rootdir,profile,subdir))) > 2:        # 2 bc the ROIs folder and .dat file are going to exist (fixed this downstream so eventually change back to zero)
                            
                            # Create flowfield
                            self.snow_directory = os.path.join(rootdir,profile,subdir)
                            self.full_flowfield = Flowfield_PIV_Full(self.snow_directory)
                            
                            # Calculate Speed (should I do this before or after calculating stats?)
                            self.flow_speed = np.sqrt((np.array(self.full_flowfield.u_flow_smooth))**2 + (np.array(self.full_flowfield.v_flow_smooth))**2)
                            
                            # Calculate Stats
                            self.flow_speed_mean = self.flow_speed.mean()
                            self.flow_speed_med = np.median(self.flow_speed)
                            self.flow_speed_std = self.flow_speed.std()
                            self.flow_speed_min = self.flow_speed.min()
                            self.flow_speed_max = self.flow_speed.max()
                            
                            file_line = [rootdir, profile, self.flow_speed_mean, self.flow_speed_med, self.flow_speed_std, self.flow_speed_min, self.flow_speed_max]
                            self.flow_summary.append(file_line)
                        
                        else:
                            self.flow_speed_mean = 'NaN'
                            self.flow_speed_med = 'NaN'
                            self.flow_speed_std = 'NaN'
                            self.flow_speed_min = 'NaN'
                            self.flow_speed_max = 'NaN'
                            
                            file_line = [rootdir, profile, self.flow_speed_mean, self.flow_speed_med, self.flow_speed_std, self.flow_speed_min, self.flow_speed_max]
                            self.flow_summary.append(file_line)
                    
                    #else: 
                    #    file_line = [rootdir, profile]
                        
            #self.flow_summary.append(file_line)
        
        # save a new csv file
        full_flow_summary = np.array(self.flow_summary)
        summary_file = 'piv_summary_output.csv'
        summary_path = os.path.join(rootdir, summary_file)
        np.savetxt(summary_path, full_flow_summary, delimiter=',', fmt='%s', header='directory, profile, mean, median, std, min, max')


# ================================================================================

# testing / run script

#test = Analysis(rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/fast_test')
piv_final = Analysis(rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/video_data/data2_20180913_extracted')




