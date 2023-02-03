
# ACW 2023

# A script to test, plot, and visualize PIV parameters and smoothing 

# Thoughts:
    # Smoothing actually looks fine and we do definitely need it -- doesnt smooth over important features like sloshing
    # There is still some noise in the actualy PIV outputs
        # small swimmers, legit noise, masking issues, ...
    # Try to remove some of the noise by adding another function in the PIV method that masks anythough outside of X SDd
    # Still a potential issue of "empty" grids having really small PIV values that aren't masked/replaced
        # not sure if this is the root of the issue yet
    # If I can remove a lot of the PIV noise I would like to get my knot spacing back to ~3-4

# =============================================================================

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.io as sio
from statistics import mean, median
from scipy import interpolate, spatial
from scipy.interpolate import UnivariateSpline, LSQUnivariateSpline
import csv
import math
import sys
import warnings

sys.path.insert(0, '/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/PIA_w_PIV')
import CTD_matching_for_PIA as ctd
from openpiv import tools, pyprocess, scaling, validation, filters
import PIV_w_Zoop_Mask_for_PIA as piv

warnings.simplefilter("ignore")

# PIV bins (5x4)
x_bins = np.array([182., 310., 438., 566., 694., 182., 310., 438., 566., 694., 182., 310., 438., 566., 694., 182., 310., 438., 566., 694.])
y_bins = np.array([517., 517., 517., 517., 517., 389., 389., 389., 389., 389., 261., 261., 261., 261., 261., 133., 133., 133., 133., 133.])

# ===========================================================================================================

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
                #flow_knt_smooth = 10                                                      # this was the setting coming into the PIV summary, I am now wondering if this is too wide, gonna play
                flow_knt_smooth = 4
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

# ===========================================================================================================

# ------------------------------------

# single frame value output and vector visualization

# sloshy flow
piv.PIV(frame1='/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1536542534/test/SHRINK-30-SPC-UW-1536542556936460-745810028220-000250.tif', frame2='/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1536542534/test/SHRINK-30-SPC-UW-1536542556986288-745810078224-000251.tif', save_setting=False, display_setting=True, verbosity_setting=True)
# uniform flow
piv.PIV(frame1='/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1501326258/test/SHRINK-30-SPC-UW-1501326269737560-158173445935-000000.tif', frame2='/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1501326258/test/SHRINK-30-SPC-UW-1501326269773942-158173495939-000001.tif', save_setting=False, display_setting=True, verbosity_setting=True)

# ---------------------------------

# vector array visualization 

# feed it a mini directory -- make a plot for every frame pair
frm_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1536951602/test'          # profile
frm_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1501326258/test'          # i think normal video (20fps)
frm_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1536542534/test'          # slow sloshing 

full_flowfield = Flowfield_PIV_Full(frm_dir)

mask_bs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

#smoothed
for i in range(len(full_flowfield.u_flow_smooth)):
    tools.display_vector_field_AW(x_bins, y_bins, full_flowfield.u_flow_smooth[i].flatten(), full_flowfield.v_flow_smooth[i].flatten(), mask_bs, scale=75, width=0.0035)

# raw
for i in range(len(full_flowfield.u_flow_smooth)):
    tools.display_vector_field_AW(x_bins, y_bins, full_flowfield.u_flow_raw[i].flatten(), full_flowfield.v_flow_raw[i].flatten(), mask_bs, scale=75, width=0.0035)

# ---------------------------------

# plot flow (raw and smooth) within a single grid (all each grid) through time

full_flowfield = Flowfield_PIV_Full('/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1501326258/shrink')
    # 590 frames and consistent flow to the left
    # some peaks around frame 100 -- its something coming through that must not be masked well (only in the corner grid it swims through)
    # otherwise I think the smoothing looks reasonable -- and its obviously needed in some places
    # some weird shit in some of the grids 
full_flowfield = Flowfield_PIV_Full('/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1536542534/shrink')
    # slow sloshing video -- in the up and down direction
    # but otherwise fairly minimal/slow flow
    # looks like the smoothing doesnt mask the sloshing and does a pretty good job
    # but there is some chaos I want to look into -- maybe plot u/v seperately 
full_flowfield = Flowfield_PIV_Full('/home/dg/Wyeth2/IN_SITU_MOTION/video_data/determined_profiles/1533711642/flow_test')
    # a section of video when the profiler is moving 
    # testing what happens when the flow is really fast

for a in range(4):
    for b in range(5):
        #print(a,b)
        x = range(len(full_flowfield.u_flow_raw))
        u_raw = full_flowfield.u_flow_raw[:,a,b]
        u_smooth = full_flowfield.u_flow_smooth[:,a,b]
        v_raw = full_flowfield.v_flow_raw[:,a,b]
        v_smooth = full_flowfield.v_flow_smooth[:,a,b]
        fig, ax = plt.subplots(2, sharex='col', sharey='row')
        ax[0].plot(x, u_raw, label = "u_raw")
        ax[0].plot(x, u_smooth, label = "u_smooth")
        ax[0].legend()
        ax[0].set(title= 'X-velocity')
        ax[1].plot(x, v_raw, label = "v_raw", color='red')
        ax[1].plot(x, v_smooth, label = "v_smooth", color='green')
        ax[1].legend()
        ax[1].set(title= 'Y-velocity')
        title = '{}_{}_{}'.format('Grid', a, b)
        fig.suptitle(title)

plt.close('all')
