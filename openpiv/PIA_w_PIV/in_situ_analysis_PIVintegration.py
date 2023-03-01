
# ACW & DG 2022

# In situ analysis:
    # reads in zoop dat files (2D motion files)
    # calculates smoothing splines and derivatives for each path 
    # calculates a PIV flowfield that is temporally interpolated
    # subtracts flowfield from each zoop path
    # Uses AI to assign a classification to each swimming path 
    # pickles each anaylsis object as it goes 

# ==========================================================

# import packages -----------------------------------

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

from openpiv import tools, pyprocess, scaling, validation, filters
import PIV_w_Zoop_Mask_for_PIA as piv
import CTD_matching_for_PIA as ctd

# set default parameters ----------------------------

warnings.simplefilter("ignore")
# numpy defaults
np.set_printoptions(suppress=True, linewidth=100)

#s_spline_flow=128       # flowfield smoothing parameter -- not using this class right now
s_spline_flow=200

verbose=False

# PIV bins (5x4)
x_bins = np.array([182., 310., 438., 566., 694., 182., 310., 438., 566., 694., 182., 310., 438., 566., 694., 182., 310., 438., 566., 694.])
y_bins = np.array([517., 517., 517., 517., 517., 389., 389., 389., 389., 389., 261., 261., 261., 261., 261., 133., 133., 133., 133., 133.])

# ==========================================================

class Path():
    """Class to pre-process (calculate smoothing splines and derivatives) a single zooplankton path for analysis
    """
    def __init__(self, path_num, path_entry, create_splines=True, k_val=None, knot_smooth=None, verbose=False):

        self.path_num=path_num
        self.path_length=path_entry['path_length']
        if self.path_length==0:
            return

        self.frames = path_entry['frames']
        self.x_pos = path_entry['B'][:,0]
        self.y_pos = path_entry['B'][:,1]

        # create empty arrays of the correct size associated with each path that can be populated in Analysis class 
        #self.x_flow_raw = np.zeros(self.x_pos.size)
        #self.y_flow_raw = np.zeros(self.y_pos.size)
        self.x_flow_smoothed = np.zeros(self.x_pos.size)
        self.y_flow_smoothed = np.zeros(self.y_pos.size)
        self.x_motion = np.zeros(self.x_pos.size)
        self.y_motion = np.zeros(self.y_pos.size)
        self.x_motion_phys = np.zeros(self.x_pos.size)
        self.y_motion_phys = np.zeros(self.y_pos.size)
        self.speed = np.zeros(self.y_pos.size)
        self.classification = np.empty(self.x_pos.size, dtype="object")
        self.classification[:] = 'None'
        self.delta_time_corrected = np.zeros(self.x_pos.size)
        self.area = np.zeros(self.x_pos.size)
        self.length = np.zeros(self.x_pos.size)
        
        if verbose:
            print('frames =',self.frames)
            print('x_pos =',self.x_pos)
            print('y_pos =',self.y_pos)
        
        if create_splines == True:
            if knot_smooth == None:
                print('Please explicitly set a value for the internal knots, knts')
                raise NameError('Halting: knot parameter knts is unspecified for LSQUnivariateSpline')
            
            knts = []
            self.knt_smooth = knot_smooth
            num_knt = int((self.frames[-1] - self.frames[0])/self.knt_smooth)
            knt_space = (self.frames[-1] - self.frames[0])/(num_knt+1)
            for k in range(num_knt):
                knts.append(knt_space*(k+1) + self.frames[0])
            #print(knts)

            # piecewise linear regression -- velocities made the most sense when derived from horizontal lines, also end points looked the best
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
            self.x_spline = LSQUnivariateSpline(self.frames, self.x_pos, knts, k=k_val)             # calculate spline for observed positions
            self.x_pos_smoothed=self.x_spline.__call__(self.frames)                                 # positions in smoothed trajectory
            self.x_vel_smoothed=self.x_spline.__call__(self.frames,1)                               # velocities in smoothed trajectory

            self.y_spline = LSQUnivariateSpline(self.frames, self.y_pos, knts, k=k_val)             # calculate spline for observed positions
            self.y_pos_smoothed=self.y_spline.__call__(self.frames)                                 # positions in smoothed trajectory
            self.y_vel_smoothed=self.y_spline.__call__(self.frames,1)                               # velocities in smoothed trajectory

# ==========================================================

def parse_paths(filename, k_value=None, knots=None, remove_zero_length=True):
    """A method to parse a dat file into a list of Path objects
    """
    ps = sio.loadmat(filename,variable_names=['data'],simplify_cells=True)['data']
    paths=[Path(i, ps[i], k_val=k_value, knot_smooth=knots, verbose=verbose) for i in range(len(ps))]
    if remove_zero_length:
        paths=[p for p in paths if p.path_length>0]
    return paths

# ==========================================================

class Flowfield_PIV_Full():
    def __init__(self, directory=None):
        """Class to create a flowfield object for an entire video using PIV anaylsis. Temportal (eventually spatial?) smoothing happens here
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
                    #self.flow_layer = self.frame_flow.output     # this is a 20 (5x4 grid) x 5 (x, y, u, v, mask) array
                    
                    #if sum(self.frame_flow.output[4]) < 12:                    # I tested this filter in Feb and it was too restrictive -- half of the videos couldnt fit a smoothing spline
                    self.flow_layer = self.frame_flow.output                    # this is a 20 (5x4 grid) x 5 (x, y, u, v, mask) array
                    #else: 
                    #    self.flow_layer.fill(np.NaN)
                        
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
                flow_knt_smooth = 6
                #flow_knt_smooth = 10
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
    
    def get_flow(self, frame_num=None, x_pos=None, y_pos=None):
        self.frame_num = (frame_num-1)  # T3D frame numbers start at 1, index needs to start at 0
        self.x_pos = x_pos
        self.y_pos = y_pos
        
        # pull the correct frame (time)
        u_point_flow = self.u_flow_smooth[self.frame_num].reshape(20,)
        v_point_flow = self.v_flow_smooth[self.frame_num].reshape(20,)
                
        #find closest x and y coordinates (space)
        x_coords = x_bins
        y_coords = y_bins
        #coordinates = list(zip(self.flowfield_full_np[0,0,:], self.flowfield_full_np[0,1,:]))      # this broke when the first array was empty 
        coordinates = list(zip(x_coords, y_coords))
        tree = spatial.KDTree(coordinates)
        pos_ind = tree.query([(self.x_pos,self.y_pos)])[1][0]
        
        #look up flow field at those coordinates
        self.point_u_flow = u_point_flow[pos_ind]
        self.point_v_flow = v_point_flow[pos_ind]
        self.point_flow = [self.point_u_flow, self.point_v_flow]

class Analysis():
    """ Class to analyze pre-processed flow and swimming paths
    """
    def __init__(self, zoop_dat_file=None, zoop_paths=None, snow_directory=None, class_file=None, CTD_dir=None, class_rows=[],remove_flow=True):
        
        # Read in zoop .dat file, extract data, create a Path object for each path, save in an array called 'zoop_paths'
        self.zoop_dat_file = zoop_dat_file
        if zoop_paths == None:
            print('Trying to load zoop_dat_file: ',zoop_dat_file)
            try:
                self.zoop_paths = parse_paths(zoop_dat_file, k_value=1, knots=3)
                print('Success')
            except:
                print('****Error in loading zoop_data_file****')
                self.zoop_paths = None

        # Store directory with frames (tif images) for PIV analysis
        print('Trying to load snow frames:' ,snow_directory)
        self.snow_directory = snow_directory
        print('Success')
        
        # Read in AI classication output (.csv file)
        if class_file != None:
            self.class_file = class_file
            print('Trying to load classification file: ', class_file)
            csv_file = open(self.class_file)
            csvreader = csv.reader(csv_file)
            class_header = next(csvreader)
            #print(class_header)
            self.class_rows = class_rows
            for row in csvreader:
                self.class_rows.append(row)
            #print(class_rows)
            csv_file.close()
            print('Success')
        
        # Read in CTD directory
        print('Trying to load CTD casts from: ', CTD_dir)
        self.CTD_dir = CTD_dir
        print('Success')
    
    def remove_flow(self):
        """ a method to compute the PIV flowfield for the entire video one time, and then remove the flow at specific time/space instances for each zooplankton localization
        """
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
    
    def assign_class_and_size(self):
        '''method to match/assign classifcations to each localization in zoop_paths
        '''
        # Streamline Classication Data
        for c in self.class_rows:
            # Pull filename 
            line = c[1]
            
            # Save frame number
            frame_tag = '_grp'                                          # frame number is listed directly before group number --  I think this is the easiest way to find it
            frame_tag_ind = line.find(frame_tag)
            frame_len= 6                                                # frame number is 6 digits long  
            frame_num = line[(frame_tag_ind-frame_len):frame_tag_ind]
            
            # Save major and minor semi-axis
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
        
        # Save first frame number (from ROIs)
        self.roi_frame_num = self.class_rows[0][10]
        #print(self.roi_frame_num)
        
        # Match ROI to Classification/Size Data
        for p in self.zoop_paths:
            for l in range(len(p.frames)):
                
                # Save frame, x, and y position of that localization
                self.frame = (p.frames[l] + (int(self.roi_frame_num)-1))                    # Need to line up frames numbers - .dat files always start at 1, ROI frames can start anywhere (usually 0)
                x_pos = p.x_pos[l]
                y_pos = p.y_pos[l]
                
                # Pull ROI infomration from frame number
                self.rois = self.np_class_rows[(self.np_class_rows[:,-5]) == self.frame, :]   # save lines of np_class_rows at correct frame
                self.roi = self.rois[(self.rois[:,-4] < (x_pos+2)) & (self.rois[:,-4] > (x_pos-2)) & (self.rois[:,-3] < (y_pos+2)) & (self.rois[:,-3] > (y_pos-2)),:]         # if the center of the ROI is within sq pixels of the localization -- match it
                
                if len(self.roi) == 1:
                    p.classification[l] = self.roi[:,4][0]
                    p.length[l] = self.roi[:,-2][0]
                    p.area[l] = self.roi[:,-1][0]
                    #print('SUCCESS: Match found')              # only printing errors right now to reduce output
                if len(self.roi) == 0:
                    print('ERROR: No match found')
                if len(self.roi) > 1:
                    print('ERROR: more than more 1 ROI found')
    
    def assign_chemistry(self):
        # extract video profile from file path
        pro_line = self.class_rows[1][1]    
        # Save frame number
        pro_tag = '-UW-'
        pro_tag_ind = pro_line.find(pro_tag)+4
        pro_len= 10
        profile_number = int(pro_line[pro_tag_ind:(pro_tag_ind+pro_len)])
        
        # calls imported script that reads in directory of raw CTD data, matches nearest CTD cast, and stores chemisty
        CTD_chemistry = ctd.Analysis(CTDdir=self.CTD_dir ,profile=profile_number)

        # assign CTD data to analysis object         
        self.profile = CTD_chemistry.vid_datnum
        self.nearest_earlier_cast = CTD_chemistry.nearest_earlier_cast
        self.time_offset = CTD_chemistry.time_offset
        self.temp_avg = CTD_chemistry.temp_avg
        self.fluor_avg = CTD_chemistry.fluor_avg
        self.depth_avg = CTD_chemistry.depth_avg
        self.salinity_avg = CTD_chemistry.salinity_avg
        self.oxygen_mgL_avg = CTD_chemistry.oxygen_mgL_avg

    def convert_to_physical(self):
        # convert motion from pixels/frame to mm/sec
        # Frame is 2600 x 3504 (actually 650 x 876 but same ratio) pixels or ~57.2mm x 77.1mm
        
        # Generate a list of timestamps and frame numbers from the original directory of tif images
        # Isolate 16 digit unix timestamps (microseconds)
        tif_list=[f for f in os.listdir(self.snow_directory) if  \
                    (os.path.isfile(os.path.join(self.snow_directory, f)) and f.endswith('.tif'))]
        tif_list.sort()
        if len(tif_list) > 0:
            # Store unix timestamp from image filename
            unix_list = []
            frame_nums = []
            for frm_file in tif_list:  
                line = str(frm_file)
                time_tag = 'UW-'
                time_tag_index = line.find(time_tag)+3
                time_len = 16
                unix = line[(time_tag_index):(time_tag_index+time_len)]
                unix_list.append(unix)
            # Store frame number from image fileman
                frame_tag = '.tif'
                frame_tag_ind = line.find(frame_tag)
                frame_len= 6
                frame_name = line[(frame_tag_ind-frame_len):frame_tag_ind]
                frame_nums.append(int(frame_name))
            # Convert unix to datetime
            datetime_list = []
            for time in unix_list:
                unix_list.sort()
                timestamp = (pd.to_datetime(int(time),unit='us')) 
                #print(datetime)
                datetime_list.append(timestamp)
        
        # save first ROI frame number
        first_roi_frame = frame_nums[0]
        
        # calculate the average frame rate for reference  
        deltas = [x - datetime_list[i - 1] for i, x in enumerate(datetime_list)][1:]
        delta_time = []
        for delta in deltas:
            delta_time.append(delta.total_seconds())
        frame_diff = [x - frame_nums[i - 1] for i, x in enumerate(frame_nums)][1:]
        delta_time_corrected = np.array(delta_time) / np.array(frame_diff)
        avg_dt = sum(delta_time_corrected) / len(delta_time_corrected)
        self.avg_frame_rt = 1/avg_dt
        
        for p in self.zoop_paths:
            # Calculate frame specific time offsets (there is large variation in frame rates between and among videos)
            for l in range(len(p.frames)-1):
                roi_frame = (p.frames[l] + (int(first_roi_frame)-1))         # Need to line up frames numbers - .dat files always start at 1, ROI frames can start anywhere (usually 0)
                ind = frame_nums.index(roi_frame)
                
                # Calculate time detla between frames
                delta = datetime_list[ind+1] - datetime_list[ind]
                # Convert from Timedelta object to seconds
                delta_time = delta.total_seconds()
                
                # Calculate different between frame numbers
                frame_diff = frame_nums[ind+1] - frame_nums[ind]
                
                # Correct time delta for the number of frames ellapsed
                p.delta_time_corrected[l] = delta_time / frame_diff
            
            # Making the last offset the same as the second to last (this might not be the best solution)
            p.delta_time_corrected[-1] = p.delta_time_corrected[-2]
            
            # time conversion -------------------------------------------------------------
            p.x_motion_phys = np.array(p.x_motion) * (1 / np.array(p.delta_time_corrected))           # frames to sec (frame rate = 1/dt)
            p.y_motion_phys = np.array(p.y_motion) * (1 / np.array(p.delta_time_corrected))           # frames to sec (frame rate = 1/dt)
            
            # space conversion ------------------------------------------------------------
            p.x_motion_phys = np.array(p.x_motion_phys) * (77.1 / 876)                                # pixels to mm 
            p.y_motion_phys = np.array(p.y_motion_phys) * (57.2 / 650)                                # pixels to mm 
            
            # calculate speed -------------------------------------------------------------
            p.speed = np.sqrt((np.array(p.x_motion_phys))**2 + (np.array(p.y_motion_phys))**2)

    def remove_flow_point(self, plot_flow_motion=True):
            """ a method to calculate the PIV flow for each frame within each zooplankton path and smooth over the path
                NOT using right now -- computationally slow
            """
            for p in self.zoop_paths:
                #for l in range(len(p.frames)-1):       # this would avoid the broken frame pair at the last frame if I wanted that
                for l in range(len(p.frames)):
                    
                    # get PIV flow 
                    self.flowfield = Flowfield_PIV_Point(p.frames[l], self.snow_directory, p.x_pos[l], p.y_pos[l])
                    # Unsmoothed PIV flow
                    p.x_flow_raw[l] = self.flowfield.point_x_flow             # this is a little confusing and maybe needs a better name because it stored in zoop_paths, but is the SNOW flow field at the time of each zoop path
                    p.y_flow_raw[l] = self.flowfield.point_y_flow
                    
                # smooth PIV flow
                # generate internal knots
                flow_knts = []
                flow_knt_smooth = 3
                flow_num_knts = int((p.frames[-1] - p.frames[0])/flow_knt_smooth)
                flow_knt_space = (p.frames[-1] - p.frames[0])/(flow_num_knts+1)
                for k in range(flow_num_knts):
                    flow_knts.append(flow_knt_space*(k+1) + p.frames[0])
                #print(flow_knts)
                
                # assign zero weight to nan values (https://gemfury.com/alkaline-ml/python:scipy/-/content/interpolate/fitpack2.py)
                wx = np.isnan(p.x_flow_raw)
                p.x_flow_raw[wx] = 0.
                wy = np.isnan(p.y_flow_raw)
                p.y_flow_raw[wy] = 0.
                
                # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
                p.x_flow_spline = LSQUnivariateSpline(p.frames, p.x_flow_raw, flow_knts, w=~wx, k=1)       # calculate spline for observed flow
                p.x_flow_smoothed = p.x_flow_spline.__call__(p.frames)                                         # flow velocity in smoothed trajectory
                p.y_flow_spline = LSQUnivariateSpline(p.frames, p.y_flow_raw, flow_knts, w=~wy, k=1)
                p.y_flow_smoothed = p.y_flow_spline.__call__(p.frames)
                
                #p.x_snow_flow = p.x_flow_smoothed
                #p.y_snow_flow = p.y_flow_smoothed
                
                # calculate motion
                p.x_motion = (p.x_vel_smoothed - p.x_flow_smoothed)
                p.y_motion = (p.y_vel_smoothed - p.y_flow_smoothed)

# ==========================================================================================

# Retired Classes

class Flowfield_PTV():
    """Class to create a flowfield object for the entire video from snow tracks using PTV anaylsis 
        NOT USING -- tracking didnt do well in high density environments
    """
    def __init__(self,paths,flowfield_type=None,frame_stat='med',use_smoothed=True,make_plot=False): 

        # get an array of the frames represented in paths
        self.all_frames=np.array([], dtype=np.uint8)
        for p in paths:
            self.all_frames=np.unique(np.concatenate((self.all_frames,p.frames)))
        #  print('all_frames = ',self.all_frames)

        # Obtain statistics for velocities in each frame
        self.frame_x_vels=np.zeros(self.all_frames.shape)
        self.frame_y_vels=np.zeros(self.all_frames.shape)
        self.frame_samples=np.zeros(self.all_frames.shape)
        for i,f in enumerate(self.all_frames):
            self.x_vels=np.array([], dtype=float)
            self.y_vels=np.array([], dtype=float)
            for p in paths:
                # get index of frame f in p, if it's present
                w=np.where(p.frames==f)
                if w[0].size>0:
                    # frame is in path
                    f_index=w[0][0] 
                    # print(f_index,p.x_vel_smoothed[f_index])
                    if use_smoothed:
                        self.x_vels=np.append(self.x_vels,p.x_vel_smoothed[f_index])
                        self.y_vels=np.append(self.y_vels,p.y_vel_smoothed[f_index])
                    else:
                        print("Haven't yet set up velocities for non-smoothed paths...")
                        raise NameError('Non-smooth veloities not yet supported')

            self.frame_samples[i]=self.x_vels.size # number of paths represented in this frame
            if frame_stat == 'avg':    # take appropriate stats
                self.frame_x_vels[i]=np.mean(self.x_vels)
                self.frame_y_vels[i]=np.mean(self.y_vels)
            elif frame_stat == 'med':
                self.frame_x_vels[i]=np.median(self.x_vels)
                self.frame_y_vels[i]=np.median(self.y_vels)
            else:
                raise NameError('Unsupported choice of velocity stats')
        
        # Execute current methods:
        self.temporal_interpolation()
        if make_plot:
            self.plot_flow_spline()

    def temporal_interpolation(self,use_weighting=True,ssf=s_spline_flow):
        '''https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html'''
        self.x_vel_temporal_spline = UnivariateSpline(self.all_frames,self.frame_x_vels, k=3,s=ssf)
        self.y_vel_temporal_spline = UnivariateSpline(self.all_frames,self.frame_y_vels, k=3,s=ssf)

        self.x_vel_temporal_eval = self.x_vel_temporal_spline(self.all_frames)
        self.y_vel_temporal_eval = self.y_vel_temporal_spline(self.all_frames)

    def plot_flow_spline(self):
        self.all_frames
        self.frame_x_vels
        self.x_vel_temporal_eval
        fig, (ax1, ax2) = plt.subplots(2)
        ax1.plot(self.all_frames,self.frame_x_vels, 'ro', self.all_frames,self.x_vel_temporal_eval, 'r')
        ax1.set_title('flowfield: x-velocity')
        ax2.plot(self.all_frames,self.frame_y_vels, 'bo', self.all_frames,self.y_vel_temporal_eval, 'b')
        ax2.set_title('flowfield: y-velocity')
        plt.show()

    def get_flow(self, frame=None):
        self.point_x_flow = self.x_vel_temporal_spline.__call__(frame) 
        self.point_y_flow = self.y_vel_temporal_spline.__call__(frame)
        self.point_flow = [self.point_x_flow, self.point_y_flow]

class Flowfield_PIV_Point():
    """Class to create a flowfield object for a specific frame pair using PIV anaylsis 
        NOT USING -- calling the PIV  analysis at specific localizations was computationally slow and didn't allow for video wide spatial/temportal smoothing
    """
    def __init__(self, frame_num=None, directory=None, x_pos=None, y_pos=None):
        self.frame_num = frame_num
        self.directory = directory
        self.x_pos = x_pos
        self.y_pos = y_pos
        
        # When frames go through Tracker3D all frames are re-numbered starting at 1
            # but the first tif image can start at a range of values
            # current_roi_frame = curent_dat_frame + first_roi_frame - 1
        # Store for ROI frame number
        tif_list=[f for f in os.listdir(self.directory) if  \
                    (os.path.isfile(os.path.join(self.directory, f)) and f.endswith('.tif'))]
        tif_list.sort()
        first_roi_frame = tif_list[0][-10:-4]
        roi_frame_a = self.frame_num + int(first_roi_frame) - 1
        roi_frame_b = self.frame_num  + int(first_roi_frame)
        #print(roi_frame_a)
        #print(roi_frame_b)
        
        roi_image_a = [f for f in os.listdir(self.directory) if  \
                    (os.path.isfile(os.path.join(self.directory, f)) and f.endswith(str("%06d"%(roi_frame_a))+'.tif'))]
        roi_image_b = [f for f in os.listdir(self.directory) if  \
                    (os.path.isfile(os.path.join(self.directory, f)) and f.endswith(str("%06d"%(roi_frame_b))+'.tif'))]
        #print(roi_image_a)
        #print(roi_image_b)
        image_a_len = len(roi_image_a)
        image_b_len = len(roi_image_b)
        
        if ((image_a_len>0) and (image_b_len>0)):                               # handles missing frames
            try:                                                                # handles corrupted frames 
                self.frame_a = os.path.join(self.directory, roi_image_a[0])
                self.frame_b = os.path.join(self.directory, roi_image_b[0])
                #print(self.frame_a)
                #print(self.frame_b)
                self.get_flow()
            except:
                print("Broken frame pair: "+str(roi_image_a)+", "+str(roi_image_b))
                self.point_x_flow = float('NaN')
                self.point_y_flow = float('NaN')
                self.point_flow = [self.point_x_flow, self.point_y_flow]
        else:
            print("Broken frame pair: "+str(roi_image_a)+", "+str(roi_image_b))
            self.point_x_flow = float('NaN')
            self.point_y_flow = float('NaN')
            self.point_flow = [self.point_x_flow, self.point_y_flow]

    def get_flow(self):
        self.frame_flow = piv.PIV(frame1=self.frame_a, frame2=self.frame_b, save_setting=False, display_setting=False, verbosity_setting=False)
        
        # find closest x and y coordinates
        coordinates = list(zip(self.frame_flow.output[0], self.frame_flow.output[1]))
        #coordinates = list(zip(self.frame_flow.output[0]*96.52, self.frame_flow.output[1]*96.52))
        tree = spatial.KDTree(coordinates)
        pos_ind = tree.query([(self.x_pos,self.y_pos)])[1][0]
        
        # look up flow field at those coordinates
        self.point_x_flow = self.frame_flow.output[2, pos_ind]
        self.point_y_flow = self.frame_flow.output[3, pos_ind]
        self.point_flow = [self.point_x_flow, self.point_y_flow]