
# ACW 2023
# 
# Script to flag profiles and create a sym link for the first 1000 frames into a different folder that I can process 

# Need to think about how I am going to flag it as a profile
    # I think I am going to start with time difference from profile
    # But this might not be perfect

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

#max_frames = 10000

# PIV bins (5x4)
x_bins = np.array([182., 310., 438., 566., 694., 182., 310., 438., 566., 694., 182., 310., 438., 566., 694., 182., 310., 438., 566., 694.])
y_bins = np.array([517., 517., 517., 517., 517., 389., 389., 389., 389., 389., 261., 261., 261., 261., 261., 133., 133., 133., 133., 133.])

#np.set_printoptions(suppress=True, linewidth=1000) 
#np.set_printoptions(threshold=sys.maxsize)
#pd.set_option('display.max_rows', None) 

# -----------------------------------------------------------------------------

# parameters

CTD_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/CTD_data/2018_DGC_fullcasts'

ctd_diff_cutoff = 75            # seconds
profile_min_frame = 2000        # number of frames

#dest_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/profile_test'
#dest_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/video_data/determined_profiles'

source_folder = 'shrink'
dest_folder = 'shrink_crop'

# batch videos ----------------------------------------------------------------
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/video_data/data2_20180913_extracted'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/fast_test'
rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/video_data/determined_profiles'

frame_dir_list = []
for profile in os.listdir(rootdir):
    if os.path.isdir(os.path.join(rootdir,profile)):
        for subdir in os.listdir(os.path.join(rootdir,profile)):
            if subdir == 'shrink':
                frame_dir_list.append(os.path.join(rootdir,profile,subdir))

# single video ----------------------------------------------------------------
#frame_dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test']
#frame_dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1536951602/shrink']       # I think its a profile
#frame_dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1501326258/shrink']
#frame_dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1534705207/shrink_test']   # profile
frame_dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1536951602/test']           #profile
#frame_dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1501326258/test']           # i think normal video (20fps)
#frame_dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/video_data/determined_profiles/1536346802/shrink']
# =============================================================================

# this function uses the .DGC file to determine how many frames to move over
# however seemed to move too many frames -- Danny says I can't trust that the profiler and camera were triggered at the same time

def Header():
    for i in range(len(CTD_chemistry.full_cast.ctd_data)):
        if CTD_chemistry.full_cast.ctd_data['Depth'][i] < (CTD_chemistry.full_cast.ctd_data['Depth'][0] - 0.1):
            return CTD_chemistry.full_cast.ctd_data[0:i]

# ============================================================================

determined_profiles = []

for frm_dir in frame_dir_list:
    
    # PART A) Generate a tiff list and extract datetime/frame rates ----------------------------
    
    # generate a list of all the .tif images in directory
    tif_list=[f for f in os.listdir(frm_dir) if  \
                (os.path.isfile(os.path.join(frm_dir, f)) and f.endswith('.tif'))]
    tif_list.sort()
        
    # Create empty arrays to store outputs
    unix_list = []
    frame_nums = []
    datetime_list = []
    
    if len(tif_list) > 0:
        # Find information from tif filename
        for frm_file in tif_list:  
            
            # Extract unix timestamp from filename
            line = str(frm_file)
            time_tag = 'UW-'
            time_tag_index = line.find(time_tag)+3
            time_len = 16
            unix = line[(time_tag_index):(time_tag_index+time_len)]
            unix_list.append(unix)
            
            # Extract frame number from filename 
            frame_tag = '.tif'
            frame_tag_ind = line.find(frame_tag)
            frame_len= 6
            frame_name = line[(frame_tag_ind-frame_len):frame_tag_ind]
            frame_nums.append(int(frame_name))
            
        # Convert unix to datetime
        for time in unix_list:
            unix_list.sort()
            #timestamp = (pd.to_datetime(int(time),unit='us'))              # !!! IMPORTANT EDIT !!!
            timestamp = pd.to_datetime(int(time),unit='us', utc=True)       # explicitly still in UTC
            timestamp = timestamp.tz_convert('US/Pacific')                  # convert to pacific - same timezone as CTD casts
            datetime_list.append(timestamp)
            
        deltas = [x - datetime_list[i - 1] for i, x in enumerate(datetime_list)][1:]
        # Convert from Timedelta object to seconds
        delta_time = []
        for delta in deltas:
            delta_time.append(delta.total_seconds())
            
        # Calculate different between frame numbers
        frame_diff = [x - frame_nums[i - 1] for i, x in enumerate(frame_nums)][1:]
        # Correct time delta for the number of frames ellapsed
        delta_time_corrected = np.array(delta_time) / np.array(frame_diff)
        avg_dt = sum(delta_time_corrected) / len(delta_time_corrected)
        avg_frame_rate = 1 / avg_dt
        
        # PART B) Call CTD script and determine time difference from cast -----------------------------------------------------------
        
        # take the first tif time stamp and crop to s (from us)
        profile_number = unix_list[0][0:10]
        CTD_chemistry = ctd.Analysis(CTDdir=CTD_dir ,profile=profile_number)
        
        # old method -----------------
        ctd_header = Header()
        scans = len(ctd_header)
        num_frames = int(scans * (1/4) * avg_frame_rate)
        # ----------------------------
        
        # PART C) Check is profile qualifcations are met and create symbolic links  -----------------------------------------------------------------------------
            
        nearest_ctd_offset = CTD_chemistry.time_offset
        print('Seconds from CTD cast: ', nearest_ctd_offset.total_seconds())
        print('Number of frames: ', len(tif_list))
        print('Frame rate: ', avg_frame_rate)
        print('Frames in header: ', num_frames)
        
        if nearest_ctd_offset.total_seconds() < ctd_diff_cutoff and len(tif_list) > profile_min_frame:
            
            # Create a list of profiles and create links for all the profiles so they are in one place
            # determined_profiles.append(frm_dir)
            # orig_profile_dir = os.path.dirname(frm_dir)
            # profile_sym_link = os.path.join(dest_dir, os.path.basename(orig_profile_dir))
            # os.symlink(orig_profile_dir, profile_sym_link)
            # print('Symbolic link created: ', profile_sym_link)
            
            # Generate flowfield and figure out where to crop/how many frames to move over
                # IN PROGRESS
            #full_flowfield = Flowfield_PIV_Full(frm_dir)
            #num_frames = 
            
            # create a new directory
            new_dir = os.path.join(os.path.dirname(frm_dir),dest_folder)
            
            if os.path.exists(new_dir):
                print('Cropped directory already exists: ', new_dir)
                print('------')
            else:
                os.mkdir(new_dir)
                print('Made new cropped directory:', new_dir)
                
                # create sym links for the first n frames from tif list 
                for tif in tif_list[0:num_frames]:
                    source = os.path.join(frm_dir, tif)
                    destination = os.path.join(new_dir, tif)
                    os.symlink(source, destination)
                print('Symbolic links created')
                print('------')
            
        else:
            print('Not considered a profile - skipped')
            print('------')






# ==================================================================================


# for profile in os.listdir(dest_dir):
    
#     # create a new directory
#     new_dir = os.path.join(dest_dir,profile,dest_folder)
    
#     if os.path.exists(new_dir):
#         print('Cropped directory already exists: ', new_dir)
#         print('------')
#     else:
#         os.mkdir(new_dir)
#         print('Made new cropped directory:', new_dir)
        
#         # create sym links for the first n frames from tif list 
#         for tif in tif_list[0:num_frames]:
#             source = os.path.join(frm_dir, tif)
#             destination = os.path.join(new_dir, tif)
#             os.symlink(source, destination)
#         print('Symbolic links created')
#         print('------')

# ===========================================================================================================

# piv class
# I dont think I want to figure this out right nowm 

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
                flow_knt_smooth = 3
                #flow_knt_smooth = 10                                                      # this was the setting coming into the PIV summary, I am now wondering if this is too wide, gonna play
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


# ------------------------------------

# testing

full_flowfield.u_flow_smooth
full_flowfield.v_flow_smooth


piv.PIV(frame1='/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1534705207/shrink_crop/SHRINK-55-SPC-UW-1534705275769315-250012798044-000011.tif', frame2='/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1534705207/shrink_crop/SHRINK-55-SPC-UW-1534705275790548-250012898052-000012.tif', save_setting=False, display_setting=True, verbosity_setting=True)

piv.PIV(frame1='/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1534705207/shrink_test2/SHRINK-55-SPC-UW-1534705277969478-250013698118-000020.tif', frame2='/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1534705207/shrink_test2/SHRINK-55-SPC-UW-1534705277976821-250013798127-000021.tif', save_setting=False, display_setting=True, verbosity_setting=True)

# ---------------------------------

#frame_dir_list = ['/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1534705207/shrink_test']   # profile
frm_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1536951602/test'          # profile
frm_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1501326258/test'          # i think normal video (20fps)
frm_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/fast_test/1536542534/test'          # slow sloshing 

full_flowfield = Flowfield_PIV_Full(frm_dir)

mask_bs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

for i in range(len(full_flowfield.u_flow_smooth)):
    tools.display_vector_field_AW(x_bins, y_bins, full_flowfield.u_flow_smooth[i].flatten(), full_flowfield.v_flow_smooth[i].flatten(), mask_bs, scale=75, width=0.0035)

for i in range(len(full_flowfield.u_flow_smooth)):
    tools.display_vector_field_AW(x_bins, y_bins, full_flowfield.u_flow_raw[i].flatten(), full_flowfield.v_flow_raw[i].flatten(), mask_bs, scale=75, width=0.0035)
    
# plot x_smooth, x_raw, y_smooth, y_raw
x = range(len(full_flowfield.u_flow_raw))
u_raw = full_flowfield.u_flow_raw[:,3,3]


