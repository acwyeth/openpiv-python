# ACW & DG 2021

# In situ analysis:
    # reads in zoop and snow dat files (2D motion files)
    # calculates smoothing splines and derivatives for each path 
    # calculates a flowfield that is temportally interpolated
    # subtracts flowfield from each zoop path


# Notes/ToDo:
# Old and new velocity arent that close -- becauce I need to USE TIME vs frames
    # look up frames per second (I think its 22)
# Look up field of view and use a real calibration file 
# Write a method that saves an output file (for each zooplankton swimming path: timestamp, path#, start_frame, #frames, x_motion, y_motion, classification)

# ==========================================================
# packages for this script
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

# packages for PIV script
#sys.path.insert(0, '/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv')
from openpiv import tools, pyprocess, scaling, validation, filters
import PIV_w_Zoop_Mask_for_PIA as piv

# paskages for CTD matching
import CTD_matching_for_PIA as ctd

# numpy defaults
np.set_printoptions(suppress=True, linewidth=100)

# default parameters

#s_spline_path=2         # path smoothing parameter
#s_spline_path=100         # path smoothing parameter

#s_spline_flow=128       # flowfield smoothing parameter
s_spline_flow=200

#knot_spacing=20

verbose=False

# ==========================================================

class Path():
    """Class to pre-process (calculate smoothing splines and derivatives) a single path for analysis
    """
    def __init__(self, path_num, path_entry, create_splines=True, k_val=None, knot_smooth=None, verbose=False):

        self.path_num=path_num
        self.path_length=path_entry['path_length']
        if self.path_length==0:
            return

        self.frames = path_entry['frames']
        self.x_pos = path_entry['B'][:,0]
        self.y_pos = path_entry['B'][:,1]

        # create empty arrays of the correct size associated with each path that can be filled in downstream
            # populated in Analysis class 
        self.x_snow_flow = np.zeros(self.x_pos.size)
        self.y_snow_flow = np.zeros(self.y_pos.size)
        self.x_motion = np.zeros(self.x_pos.size)
        self.y_motion = np.zeros(self.y_pos.size)
        self.classification = np.empty(self.x_pos.size, dtype="object")
        self.classification[:] = 'None'
        
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

            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
            self.x_spline = LSQUnivariateSpline(self.frames, self.x_pos, knts, k=k_val)             # calculate spline for observed positions
            self.x_pos_smoothed=self.x_spline.__call__(self.frames)                                 # positions in smoothed trajectory
            self.x_vel_smoothed=self.x_spline.__call__(self.frames,1)                               # velocities in smoothed trajectory

            self.y_spline = LSQUnivariateSpline(self.frames, self.y_pos, knts, k=k_val)             # calculate spline for observed positions
            self.y_pos_smoothed=self.y_spline.__call__(self.frames)                                 # positions in smoothed trajectory
            self.y_vel_smoothed=self.y_spline.__call__(self.frames,1)                               # velocities in smoothed trajectory

# ==========================================================

def parse_paths(filename, k_value=None, knots=None, remove_zero_length=True):
    # A method to parse a dat file into a list of Path objects
    ps = sio.loadmat(filename,variable_names=['data'],simplify_cells=True)['data']
    paths=[Path(i, ps[i], k_val=k_value, knot_smooth=knots, verbose=verbose) for i in range(len(ps))]
    if remove_zero_length:
        paths=[p for p in paths if p.path_length>0]
    return paths

# ==========================================================

# using PTV analysis (not using right now)
class Flowfield():
    """Class to create a flowfield object for the entire video (spatial and temporal interpolations)
    """
    def __init__(self,paths,flowfield_type=None,frame_stat='med',use_smoothed=True,make_plot=False): 

        # get an array of the frames represented in paths
        self.all_frames=np.array([], dtype=np.uint8)
        for p in paths:
            self.all_frames=np.unique(np.concatenate((self.all_frames,p.frames)))
        #  print('all_frames = ',self.all_frames)

        # Obtain statistics for velocities in each frame ------ ## This section is a little confusing to me
        self.frame_x_vels=np.zeros(self.all_frames.shape)
        self.frame_y_vels=np.zeros(self.all_frames.shape)
        self.frame_samples=np.zeros(self.all_frames.shape)
        for i,f in enumerate(self.all_frames):
            self.x_vels=np.array([], dtype=float)
            self.y_vels=np.array([], dtype=float)
            for p in paths:
                # get index of frame f in p, if it's present
                w=np.where(p.frames==f)
                # print('w = ',w)
                # print('w[0] = ',w[0])
                # print('w[0].size = ',w[0].size)
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
        #print('frame_samples = ',self.frame_samples)
        #print('frame_x_vels = ',self.frame_x_vels)
        #print('frame_y_vels = ',self.frame_y_vels)

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

class Flowfield_PIV():
    def __init__(self, frame_num=None, directory=None, x_pos=None, y_pos=None):
        self.frame_num = frame_num
        self.directory = directory
        self.x_pos = x_pos
        self.y_pos = y_pos
        
        # When frames go through Tracker3D all frames are re-numbered starting at 1
        frame_ind = self.frame_num - 1
        
        # Find frames 
        tif_list=[f for f in os.listdir(self.directory) if  \
                    (os.path.isfile(os.path.join(self.directory, f)) and f.endswith('.tif'))]
        tif_list.sort()
        self.frame_a = os.path.join(directory, tif_list[frame_ind])
        self.frame_b = os.path.join(directory, tif_list[frame_ind+1])
        
        self.get_flow()
    
    def get_flow(self):
        #try:
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
        
        # except:
        #    print('frame_a and/or frame_b corrupt')

class Analysis():
    """ Class to analyze pre-processed flow and swimming paths
    """
    def __init__(self, zoop_dat_file=None, zoop_paths=None, snow_directory=None, class_file=None, CTD_dir=None, class_rows=[],remove_flow=True):
        # Read in zoop .dat file, extract data, create a Path object for each path, save in an array called 'paths'
        self.zoop_dat_file = zoop_dat_file
        if zoop_paths == None:
            print('Trying to load zoop_dat_file: ',zoop_dat_file)
            try:
                self.zoop_paths = parse_paths(zoop_dat_file, k_value=1, knots=3)
                print('Success')
            except:
                print('****Error in loading zoop_data_file****')
                self.zoop_paths = None

        # Store directory with frames for PIV analysis
        print('Trying to load snow frames:' ,snow_directory)
        self.snow_directory = snow_directory
        print('Success')
        
        # Read in classication output
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
    
    def remove_flow(self, plot_flow_motion=True):
        for p in self.zoop_paths:
            for l in range(len(p.frames)-1):
            #self.flowfield.get_flow(p.frames)
                self.flowfield = Flowfield_PIV(p.frames[l], self.snow_directory, p.x_pos[l], p.y_pos[l])
                
                p.x_snow_flow[l] = self.flowfield.point_x_flow             # this is a little confusing and maybe needs a better name because it stored in zoop_paths, but is the SNOW flow field at the time of each zoop path
                p.y_snow_flow[l] = self.flowfield.point_y_flow
                #print('Flow=', self.flowfield.point_x_flow)
                
                # Right now subtracting a non-smoothed flow from a smoothing swimming path -- doesnt feel right but getting it to work
                p.x_motion = (p.x_vel_smoothed - p.x_snow_flow)
                p.y_motion = (p.y_vel_smoothed - p.y_snow_flow)
    
    def assign_classification(self):
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
        # Convert to numpy array
        self.np_class_rows = np.array(self.class_rows, dtype=object)

        # Match ROI to Classification Data
        for p in self.zoop_paths:
            for l in range(len(p.frames)):
                # Save frame, x, and y position of that localization
                frame = (p.frames[l]-1)
                #frame = (p.frames[l]+499)                                             # TEMP FIX: for motion test video
                #frame = (p.frames[l]+599) 
                x_pos = p.x_pos[l]
                y_pos = p.y_pos[l]
                # Pull ROI infomration from frame number
                rois = self.np_class_rows[(self.np_class_rows[:,-3]) == frame, :]   # save lines of np_class_rows at correct frame
                roi = rois[(rois[:,-2] < (x_pos+2)) & (rois[:,-2] > (x_pos-2)) & (rois[:,-1] < (y_pos+2)) & (rois[:,-1] > (y_pos-2)),:]         # if the center of the ROI is within sq pixels of the localization -- match it
                if len(roi) == 1:
                    p.classification[l] = roi[:,4][0]
                    print('SUCCESS: Match found')
                if len(roi) == 0:
                    print('ERROR: No match found')
                if len(roi) > 1:
                    print('ERROR: more than more 1 ROI found')

    def assign_chemistry(self):
        # extract video profile from file path
        pro_line = self.class_rows[1][1]    
        # Save frame number
        pro_tag = '-UW-'                                          # frame number is listed directly before group number --  I think this is the easiest way to find it
        pro_tag_ind = pro_line.find(pro_tag)+4
        pro_len= 10                                                # frame number is 6 digits long  
        profile_number = int(pro_line[pro_tag_ind:(pro_tag_ind+pro_len)])
        
        CTD_chemistry = ctd.Analysis(CTDdir=self.CTD_dir ,profile=profile_number)

        # assign CTD data to analysis object         
        self.profile = CTD_chemistry.vid_datnum
        self.nearest_earlier_cast = CTD_chemistry.nearest_earlier_cast
        #self.time_offset = CTD_chemistry.time_offset
        self.temp_avg = CTD_chemistry.temp_avg
        self.fluor_avg = CTD_chemistry.fluor_avg
        self.depth_avg = CTD_chemistry.depth_avg
        self.salinity_avg = CTD_chemistry.salinity_avg
        self.oxygen_mgL_avg = CTD_chemistry.oxygen_mgL_avg
        
    #def write_zoop_file(self):
        # video ID (datenum)
        # path number
        # start frame
        # number of points
        # avg x motion
        # avg y motion
        # most common (?) classification
        # classification flag (something to indicate if <75% of classifications matched or something)

    def plot_motion(self, plot_flow_motion=False, plot_flow_position=False):
        # plot zooplankton MOTION
        if plot_flow_motion:
            for p in self.zoop_paths:   
                plt.scatter(p.x_motion, p.y_motion)
                plt.plot(p.x_motion, p.y_motion)
                #plt.plot(p.x_motion, p.y_motion, 'ro')          
            plt.xlabel("x-motion")
            plt.ylabel("y-motion")
            plt.axhline(y=0, color='k', linestyle='-')
            plt.axvline(x=0, color='k', linestyle='-')
            plt.show()
        
        # plot zooplankton positions
        if plot_flow_position:
            for p in self.zoop_paths:   
                plt.scatter(p.x_pos, p.y_pos)
                plt.plot(p.x_pos, p.y_pos)
                #plt.plot(p.x_motion, p.y_motion, 'ro')          
            plt.xlabel("x-pos")
            plt.ylabel("y-pos")
            plt.show()

    def check_flow(self, plot_error=False):
        for p in self.snow_paths:
            self.flowfield.get_flow(p.frames)
            p.x_snow_flow = self.flowfield.point_x_flow 
            p.y_snow_flow = self.flowfield.point_y_flow
            p.x_motion = (p.x_vel_smoothed - p.x_snow_flow)
            p.y_motion = (p.y_vel_smoothed - p.y_snow_flow)
        
        if plot_error:
            # plotting residual between snow motion and flowfield
            for p in self.snow_paths:   
                plt.scatter(p.x_motion, p.y_motion)
                plt.plot(p.x_motion, p.y_motion)
                #plt.plot(p.x_motion, p.y_motion, 'ro')          
            plt.xlabel("x-motion")
            plt.ylabel("y-motion")
            plt.axhline(y=0, color='k', linestyle='-')
            plt.axvline(x=0, color='k', linestyle='-')
            plt.show()

    def plot_flow_position(self):
        for p in self.snow_paths:   
            plt.scatter(p.x_pos, p.y_pos)
            plt.plot(p.x_pos, p.y_pos)
        plt.xlabel("x-pos")
        plt.ylabel("y-pos")
        plt.axhline(y=100, color='k', linestyle='-')
        plt.axhline(y=200, color='k', linestyle='-')
        plt.axhline(y=300, color='k', linestyle='-')
        plt.axhline(y=400, color='k', linestyle='-')
        plt.axhline(y=500, color='k', linestyle='-')
        plt.axhline(y=600, color='k', linestyle='-')
        plt.axvline(x=100, color='k', linestyle='-')
        plt.axvline(x=200, color='k', linestyle='-')
        plt.axvline(x=300, color='k', linestyle='-')
        plt.axvline(x=400, color='k', linestyle='-')
        plt.axvline(x=500, color='k', linestyle='-')
        plt.axvline(x=600, color='k', linestyle='-')
        plt.axvline(x=700, color='k', linestyle='-')
        plt.axvline(x=800, color='k', linestyle='-')
        plt.show()
