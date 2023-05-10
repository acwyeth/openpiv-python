
# a script to read in pickled video analysis objects using a look up table, unpickle, and organize final data

# For COPEPODS
# 3 swimming behaviors:
    # Drifting < 3 mm/s
    # Cruising
    # Jumping > 100 mm/s

# ACW

# 25 Aug 2022

# Execute:
    # cd Wyeth2/GIT_repos_insitu/GIT_in_situ_motion

# ==================================================================

import os 
import numpy as np
import pickle
import statistics
from collections import Counter
from datetime import *
import pandas as pd

# ==================================================================

class Path():
    """A class to organize swimming data on the path level
    """
    def __init__(self, path=None, ID=None):
        # DEFINE VARIABLES ------------
        #descriptive info
        self.path_classifications = path.classification
        self.path_areas = path.area
        self.path_heights = path.height
        self.path_widths = path.width
        self.path_length = len(path.frames)
        
        self.path_ID = ID
        self.path_classification = None
        self.path_avg_height = None
        self.path_max_height = None
        self.path_avg_width = None
        self.path_max_width = None
        self.path_avg_length = None
        self.path_max_length = None
        self.path_avg_area = None
        self.path_max_area = None
        
        #speeds
        self.all_speeds = path.speed
        self.path_jump_speeds = []
        self.path_avg_jump_speed = None
        self.path_jumps = 0
        self.frac_jumps_frames = None
        
        self.path_cruise_speeds = []
        self.path_avg_cruise_speed = None
        self.cruise_frames = 0
        
        self.path_drift_speeds = []
        self.path_avg_drift_speed = None
        self.drift_frames = 0
        
        #transitions
        self.speed_states = []
        self.trans_drift_cruise = 0
        self.trans_drift_jump = 0
        self.trans_cruise_drift = 0
        self.trans_cruise_jump = 0
        self.trans_jump_drift = 0
        self.trans_jump_cruise = 0
        
        self.trans_matrix = None
        self.cc_prob = None
        self.cd_prob = None
        self.cj_prob = None
        self.dc_prob = None
        self.dd_prob = None
        self.dj_prob = None
        self.jc_prob = None
        self.jd_prob = None
        self.jj_prob = None
        
        # FILL VARIBALES --------------
        # convert to mm (11.36 pixels = 1 mm)
        self.path_avg_height = (statistics.mean(self.path_heights) / 11.36)
        self.path_max_height = (self.path_heights.max() / 11.36)
        
        self.path_avg_width = (statistics.mean(self.path_widths) / 11.36)
        self.path_max_width = (self.path_widths.max() / 11.36)
        
        if self.path_avg_height > self.path_avg_width:
            self.path_avg_length = self.path_avg_height
        else:
            self.path_avg_length = self.path_avg_width
        
        if self.path_max_height > self.path_max_width:
            self.path_max_length = self.path_max_height
        else:
            self.path_max_length = self.path_max_width
        
        self.path_avg_area = (statistics.mean(self.path_areas) / (11.36**2))
        self.path_max_area = (self.path_areas.max() / (11.36**2))
        
        for l in range(self.path_length):
            if path.speed[l] > 100:
                self.speed_states.append('J')
                self.path_jumps = self.path_jumps + 1
                self.path_jump_speeds.append(path.speed[l])
            elif path.speed[l] < 3:                                     # drifting speed needs some more thought -- basically an estimate of flow removal error 
                self.speed_states.append('D')
                self.drift_frames = self.drift_frames + 1
                self.path_drift_speeds.append(path.speed[l])
            else:
                self.speed_states.append('C')
                self.cruise_frames = self.cruise_frames + 1 
                self.path_cruise_speeds.append(path.speed[l])
                        
        #print(path.speed)
        #print(self.speed_states_model)
        
        if len(self.path_cruise_speeds) > 0:
            self.path_avg_cruise_speed = statistics.mean(self.path_cruise_speeds)
        else:
            self.path_avg_cruise_speed = 'NaN'
        
        if len(self.path_jump_speeds) > 0:
            self.path_avg_jump_speed = statistics.mean(self.path_jump_speeds)
        else:
            self.path_avg_jump_speed = 'NaN'
        
        if self.path_length > 0:
            self.frac_jumps_frames = self.path_jumps / self.path_length
        
        if len(self.path_drift_speeds) > 0:
            self.path_avg_drift_speed = statistics.mean(self.path_drift_speeds)
        else:
            self.path_avg_drift_speed = 'NaN'
        
        # Calculate transition states! 
        for l in range(self.path_length-1):
            if self.speed_states[l] == 'D' and self.speed_states[l+1] == 'C':
                self.trans_drift_cruise = self.trans_drift_cruise + 1
            elif self.speed_states[l] == 'D' and self.speed_states[l+1] == 'J':
                self.trans_drift_jump = self.trans_drift_jump + 1
            elif self.speed_states[l] == 'C' and self.speed_states[l+1] == 'D':
                self.trans_cruise_drift = self.trans_cruise_drift + 1
            elif self.speed_states[l] == 'C' and self.speed_states[l+1] == 'J':
                self.trans_cruise_jump = self.trans_cruise_jump + 1
            elif self.speed_states[l] == 'J' and self.speed_states[l+1] == 'D':
                self.trans_jump_drift = self.trans_jump_drift + 1
            elif self.speed_states[l] == 'J' and self.speed_states[l+1] == 'C':
                self.trans_jump_cruise = self.trans_jump_cruise + 1
            
        # Calculate Swimming State Propabilities
        self.trans_matrix = self.transition_matrix(self.speed_states)
        self.cc_prob = self.trans_matrix[0,0]
        self.cd_prob = self.trans_matrix[0,1]
        self.cj_prob = self.trans_matrix[0,2]
        self.dc_prob = self.trans_matrix[1,0]
        self.dd_prob = self.trans_matrix[1,1]
        self.dj_prob = self.trans_matrix[1,2]
        self.jc_prob = self.trans_matrix[2,0]
        self.jd_prob = self.trans_matrix[2,1]
        self.jj_prob = self.trans_matrix[2,2]
    
    def transition_matrix(self, transitions):
        # Markov Transition Matrix 
        # https://stackoverflow.com/questions/46657221/generating-markov-transition-matrix-in-python -- in comments
        
        df = pd.DataFrame(transitions)
        
        # create a new column with data shifted one space
        df['shift'] = df[0].shift(-1)
        # add a count column (for group by function)
        df['count'] = 1
        
        # create dummy rows for each transition so the matrix is complete even if there isn't every transition in each path
        df.loc[len(df.index)] = ['C', 'C', 0] 
        df.loc[len(df.index)] = ['C', 'D', 0] 
        df.loc[len(df.index)] = ['C', 'J', 0] 
        df.loc[len(df.index)] = ['J', 'C', 0] 
        df.loc[len(df.index)] = ['J', 'D', 0] 
        df.loc[len(df.index)] = ['J', 'J', 0] 
        df.loc[len(df.index)] = ['D', 'C', 0] 
        df.loc[len(df.index)] = ['D', 'D', 0] 
        df.loc[len(df.index)] = ['D', 'J', 0] 
        
        # groupby and then unstack, fill the zeros
        trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
        
        # remove each of the dummy counts
        trans_mat = trans_mat - 1 
        
        # normalise by occurences and save values to get transition matrix
        trans_matrix_final = trans_mat.div(trans_mat.sum(axis=1), axis=0).values
        # convert NaNs to 0
        trans_matrix_final[np.isnan(trans_matrix_final)] = 0
        
        return trans_matrix_final 
            # NOTES ON OUTPUT
                #   C D J
                # C
                # D
                # J

class Video():
    """ A class to organize swimming data on the video level 
    """
    def __init__(self, video=None, profile=None, group=None, zoop_class=None):
        self.profile = profile
        self.group = group
        self.num_all_paths = len(video.zoop_paths)
        self.paths_of_interest = []
        self.total_frames = 0
        self.vid_jumps = 0
        self.jumps_per_path = []
        self.vid_cruise_speed = []
        self.vid_avg_cruise_speed = None
        self.paths_w_jumps = 0
        self.frac_jumps_paths = None
        self.avg_jumps_per_path = None
        
        # sort for paths of interest (right classification)
        path_id_counter = 0
        for path in video.zoop_paths:
            path_id_counter = path_id_counter + 1
            if not np.isnan(path.x_flow_smoothed).any():                                                                # skip paths with broken smoothing (for now)
                # Classification filter
                if self.most_frequent(path.classification) == zoop_class:                                              # only grab paths that are most freqently IDed as copepods
                #if self.classification_determination(List=path.classification, classification=zoop_class, thresh=0.25) == zoop_class:       # NEEDS TESTING
                    self.paths_of_interest.append(Path(path=path, ID=path_id_counter))                                                      # ONLY paths that meet these qualifiers will end up in self.paths_of_interest 
                    
        if len(self.paths_of_interest) > 0:
            for p in self.paths_of_interest:
                
                self.total_frames = self.total_frames + p.path_length                       # count the numbers of frames in paths of interest throughout the video
                
                self.vid_jumps = self.vid_jumps + p.path_jumps                              # count the total number of jumps throughout the video 
                
                self.jumps_per_path.append(p.path_jumps)
                
                if p.path_avg_cruise_speed != 'NaN':
                    #self.vid_cruise_speed.append(p.path_cruise_speeds)
                    self.vid_cruise_speed.extend(p.path_cruise_speeds)
                
                if p.path_jumps > 0:
                    self.paths_w_jumps = self.paths_w_jumps + 1                             # count how many of the paths contain at least one jump
                
            if len(self.vid_cruise_speed) > 0:
                self.vid_avg_cruise_speed = statistics.mean(self.vid_cruise_speed)          # average cruiseing speed over the whole video (from instantaneous speeds)
            else:
                self.vid_avg_cruise_speed = 'NaN'
                
            self.frac_jumps_paths = self.paths_w_jumps / len(self.paths_of_interest)        # fraction of paths in the video that contain at least one jump
            
            self.avg_jumps_per_path = statistics.mean(self.jumps_per_path)                  # average number of jumps per path in the video
    
    def most_frequent(self, List):
        # function to return the most common ROI classification from a path
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]
    
    def classification_determination(self, List, classification, thresh):
        # function to determine if path is cope/amph if above a specified threshold
        # doesnt need to be most frequent
        count = 0
        for i in List:
            if i == classification:
                count = count + 1
        if count/len(List) > thresh:
            return classification

class Group():
    """ A class to organize swimming data on the chemical/physical grouping level
    """
    def __init__(self, group_vids=None, group=None):
        self.group_vids = group_vids
        self.group = group
        self.total_vids = len(self.group_vids)
        self.total_paths = 0
        self.total_frames = 0
        self.paths_per_vid = []
        self.group_jumps = 0
        self.med_paths_per_vid = None
        self.group_cruise_speed = []                            # list of all the instantensous velocities 
        self.avg_jumps_per_path = []
        self.overall_avg_jumps_per_path = None
        self.group_avg_jump_per_frame = None
        self.group_avg_cruise_speed = None
        self.vids_w_jumps = 0
        self.frac_jumps_vids = None
        
        self.group_speed_states = []
        self.trans_matrix = []
        self.trans_matrix_prob = []
        
        self.group_speed_states1 = []
        self.group_speed_states2 = []
        self.group_speed_states3 = []
        self.group_speed_states4 = []
        self.group_speed_states10 = []
        self.trans_matrix1 = []
        self.trans_matrix_prob1 = []
        self.trans_matrix2 = []
        self.trans_matrix_prob2 = []
        self.trans_matrix3 = []
        self.trans_matrix_prob3 = []
        self.trans_matrix4 = []
        self.trans_matrix_prob4 = []
        self.trans_matrix10 = []
        self.trans_matrix_prob10 = []
        
        if self.total_vids > 0: 
            for v in self.group_vids:
                # calculate total paths and frames
                self.total_paths = self.total_paths + len(v.paths_of_interest)
                self.total_frames = self.total_frames + v.total_frames
                self.paths_per_vid.append(len(v.paths_of_interest))
                
                # calculate total jumps
                self.group_jumps = self.group_jumps + v.vid_jumps
                
                # generate an array for all the cruise speeds 
                if v.vid_avg_cruise_speed != 'NaN' and v.vid_avg_cruise_speed != None:
                    self.group_cruise_speed.extend(v.vid_cruise_speed)
                
                # generate some jump stats (number of videos in group that contain at least one jump and an array for average jumps)
                if v.vid_jumps > 0:
                    self.vids_w_jumps = self.vids_w_jumps + 1
                    self.avg_jumps_per_path.append(v.avg_jumps_per_path)
            
            # GROUP STATS -------------------------------------------------------
            # average paths per vid
            if len(self.paths_per_vid) > 0:
                self.med_paths_per_vid = statistics.median(self.paths_per_vid)
            
            # average cruise speed in the group (from instantaneous velocities)
            if len(self.group_cruise_speed) > 0:
                self.group_avg_cruise_speed = statistics.mean(self.group_cruise_speed)
            else:
                self.group_avg_cruise_speed = 'NaN'
            
            # fraction of videos that have a jump
            self.frac_jumps_vids = self.vids_w_jumps / self.total_vids
            
            # the average number of jumpas per FRAME
            if self.total_frames > 0:
                self.group_avg_jump_per_frame =  self.group_jumps / self.total_frames
            else:
                self.group_avg_jump_per_frame = 'NaN'
            
            # overall average (avg of avg) of the number of jumps in each path in group
            if len(self.avg_jumps_per_path) > 0:
                self.overall_avg_jumps_per_path = statistics.mean(self.avg_jumps_per_path)      
            else:
                self.overall_avg_jumps_per_path = 'NaN'
            
            # GROUP TRANSITION MATRIX
            for v in self.group_vids:
                for p in v.paths_of_interest:
                    self.group_speed_states.append(p.speed_states)      # all speed states from each path of interest for each group
            df_total = pd.DataFrame()
            for p in self.group_speed_states:
                df = pd.DataFrame(p)
                df['shift'] = df[0].shift(-1)
                df['count'] = 1
                df_total = pd.concat([df_total, df])
            self.trans_matrix = df_total.groupby([0, 'shift']).count().unstack().fillna(0)
            self.trans_matrix_prob = self.trans_matrix.div(self.trans_matrix.sum(axis=1), axis=0).values
            
            # GROUP TRANSITION MATRIX -- binned by size
                # this is a little hacky bc size bins aren't build into this stage of the anaylysis -- dont want to rework it for just this metric
            for v in self.group_vids:
                for p in v.paths_of_interest:
                    if p.path_avg_length <= 1:
                        self.group_speed_states1.append(p.speed_states)
                    elif p.path_avg_length > 1 and p.path_avg_length <=2:
                        self.group_speed_states2.append(p.speed_states)
                    elif p.path_avg_length > 2 and p.path_avg_length <=3:
                        self.group_speed_states3.append(p.speed_states)
                    elif p.path_avg_length > 3 and p.path_avg_length <=4:
                        self.group_speed_states4.append(p.speed_states)
                    elif p.path_avg_length > 4 and p.path_avg_length <=10:
                        self.group_speed_states10.append(p.speed_states)
                        
            #if len(self.group_speed_states1) > 0:
            df_total = pd.DataFrame()
            for p in self.group_speed_states1:
                df = pd.DataFrame(p)
                df['shift'] = df[0].shift(-1)
                df['count'] = 1
                df_total = pd.concat([df_total, df])
            
            df_total.loc[len(df_total.index)] = ['C', 'C', 0]
            df_total.loc[len(df_total.index)] = ['C', 'D', 0]
            df_total.loc[len(df_total.index)] = ['C', 'J', 0]
            df_total.loc[len(df_total.index)] = ['J', 'C', 0]
            df_total.loc[len(df_total.index)] = ['J', 'D', 0]
            df_total.loc[len(df_total.index)] = ['J', 'J', 0]
            df_total.loc[len(df_total.index)] = ['D', 'C', 0]
            df_total.loc[len(df_total.index)] = ['D', 'D', 0]
            df_total.loc[len(df_total.index)] = ['D', 'J', 0]
            self.trans_matrix1 = df_total.groupby([0, 'shift']).count().unstack().fillna(0)
            self.trans_matrix1 = self.trans_matrix1 - 1
            self.trans_matrix_prob1 = self.trans_matrix1.div(self.trans_matrix1.sum(axis=1), axis=0).values
            self.trans_matrix_prob1[np.isnan(self.trans_matrix_prob1)] = 0
                
            #if len(self.group_speed_states2) > 0:
            df_total = pd.DataFrame()
            for p in self.group_speed_states2:
                df = pd.DataFrame(p)
                df['shift'] = df[0].shift(-1)
                df['count'] = 1
                df_total = pd.concat([df_total, df])
            
            df_total.loc[len(df_total.index)] = ['C', 'C', 0]
            df_total.loc[len(df_total.index)] = ['C', 'D', 0]
            df_total.loc[len(df_total.index)] = ['C', 'J', 0]
            df_total.loc[len(df_total.index)] = ['J', 'C', 0]
            df_total.loc[len(df_total.index)] = ['J', 'D', 0]
            df_total.loc[len(df_total.index)] = ['J', 'J', 0]
            df_total.loc[len(df_total.index)] = ['D', 'C', 0]
            df_total.loc[len(df_total.index)] = ['D', 'D', 0]
            df_total.loc[len(df_total.index)] = ['D', 'J', 0]
            self.trans_matrix2 = df_total.groupby([0, 'shift']).count().unstack().fillna(0)
            self.trans_matrix2 = self.trans_matrix2 - 1
            self.trans_matrix_prob2 = self.trans_matrix2.div(self.trans_matrix2.sum(axis=1), axis=0).values
            self.trans_matrix_prob2[np.isnan(self.trans_matrix_prob2)] = 0
            
            #if len(self.group_speed_states3) > 0:
            df_total = pd.DataFrame()
            for p in self.group_speed_states3:
                df = pd.DataFrame(p)
                df['shift'] = df[0].shift(-1)
                df['count'] = 1
                df_total = pd.concat([df_total, df])
            
            df_total.loc[len(df_total.index)] = ['C', 'C', 0]
            df_total.loc[len(df_total.index)] = ['C', 'D', 0]
            df_total.loc[len(df_total.index)] = ['C', 'J', 0]
            df_total.loc[len(df_total.index)] = ['J', 'C', 0]
            df_total.loc[len(df_total.index)] = ['J', 'D', 0]
            df_total.loc[len(df_total.index)] = ['J', 'J', 0]
            df_total.loc[len(df_total.index)] = ['D', 'C', 0]
            df_total.loc[len(df_total.index)] = ['D', 'D', 0]
            df_total.loc[len(df_total.index)] = ['D', 'J', 0]
            self.trans_matrix3 = df_total.groupby([0, 'shift']).count().unstack().fillna(0)
            self.trans_matrix3 = self.trans_matrix3 - 1
            self.trans_matrix_prob3 = self.trans_matrix3.div(self.trans_matrix3.sum(axis=1), axis=0).values
            self.trans_matrix_prob3[np.isnan(self.trans_matrix_prob3)] = 0
            
            #if len(self.group_speed_states4) > 0:
            df_total = pd.DataFrame()
            for p in self.group_speed_states4:
                df = pd.DataFrame(p)
                df['shift'] = df[0].shift(-1)
                df['count'] = 1
                df_total = pd.concat([df_total, df])
            
            df_total.loc[len(df_total.index)] = ['C', 'C', 0]
            df_total.loc[len(df_total.index)] = ['C', 'D', 0]
            df_total.loc[len(df_total.index)] = ['C', 'J', 0]
            df_total.loc[len(df_total.index)] = ['J', 'C', 0]
            df_total.loc[len(df_total.index)] = ['J', 'D', 0]
            df_total.loc[len(df_total.index)] = ['J', 'J', 0]
            df_total.loc[len(df_total.index)] = ['D', 'C', 0]
            df_total.loc[len(df_total.index)] = ['D', 'D', 0]
            df_total.loc[len(df_total.index)] = ['D', 'J', 0]
            self.trans_matrix4 = df_total.groupby([0, 'shift']).count().unstack().fillna(0)
            self.trans_matrix4 = self.trans_matrix4 - 1
            self.trans_matrix_prob4 = self.trans_matrix4.div(self.trans_matrix4.sum(axis=1), axis=0).values
            self.trans_matrix_prob4[np.isnan(self.trans_matrix_prob4)] = 0
            
            if len(self.group_speed_states10) > 0:
                df_total = pd.DataFrame()
                for p in self.group_speed_states10:
                    df = pd.DataFrame(p)
                    df['shift'] = df[0].shift(-1)
                    df['count'] = 1
                    df_total = pd.concat([df_total, df])
                
                df_total.loc[len(df_total.index)] = ['C', 'C', 0]
                df_total.loc[len(df_total.index)] = ['C', 'D', 0]
                df_total.loc[len(df_total.index)] = ['C', 'J', 0]
                df_total.loc[len(df_total.index)] = ['J', 'C', 0]
                df_total.loc[len(df_total.index)] = ['J', 'D', 0]
                df_total.loc[len(df_total.index)] = ['J', 'J', 0]
                df_total.loc[len(df_total.index)] = ['D', 'C', 0]
                df_total.loc[len(df_total.index)] = ['D', 'D', 0]
                df_total.loc[len(df_total.index)] = ['D', 'J', 0]
                self.trans_matrix10 = df_total.groupby([0, 'shift']).count().unstack().fillna(0)
                self.trans_matrix10 = self.trans_matrix10 - 1
                self.trans_matrix_prob10 = self.trans_matrix10.div(self.trans_matrix10.sum(axis=1), axis=0).values
                self.trans_matrix_prob10[np.isnan(self.trans_matrix_prob10)] = 0
            else:
                self.trans_matrix_prob10 = pd.DataFrame(np.nan, index=[0, 1, 2], columns=['0', 'shift','count']).values
                
    def HHM(self):
        """ A method to format paths within each group and fit the best hidden markov model
        """
        

class Analysis():
    """ A class to read in a directory of pickled videos, sort them by designated chemical/physical conditions, and store swimming data on the path, video, and group level
    """
    def __init__(self, rootdir=None, lookup_file=None, group_method=None, oxygen_thresh=None, time_thresh1=None, time_thresh2=None, depth_thresh=None, classifier=None, save=True, output_file=None, metadata_file=None, markov_file=None):
        self.rootdir = rootdir
        self.lookup_table = np.genfromtxt(os.path.join(self.rootdir,lookup_file), dtype = str, delimiter=',', skip_header=0)
        self.video_dic = {}
        self.oxygen_thresh = oxygen_thresh
        self.time_thresh1 = time_thresh1
        self.time_thresh2 = time_thresh2
        self.depth_thresh = depth_thresh
        self.classifier = classifier
        self.output_file = output_file
        self.metadata_file = metadata_file
        self.markov_file = markov_file
        
        # 1) Read in pickled video analyses 
        for file in os.listdir(self.rootdir):
            if file.endswith('.pickle'):
                print(str(file)[0:10])
                pickle_file = open(os.path.join(self.rootdir,file),"rb")
                print(pickle_file)
                self.video_dic[str(file)[0:10]] =  pickle.load(pickle_file)
                pickle_file.close()
                
        self.keys_list = list(self.video_dic)
        
        # 2) Sort videos into different chemical/physical groups 
        self.sorted_videos = []
        for vid in self.lookup_table:
            if group_method == 'A':
                self.sort_vids_A(vid_dic=self.video_dic, video=vid, oxygen_thres=self.oxygen_thresh)
            if group_method == 'B':
                self.sort_vids_B(vid_dic=self.video_dic, video=vid, oxygen_thres=self.oxygen_thresh, time_thresh1=self.time_thresh1, time_thresh2=self.time_thresh2, depth_thresh=self.depth_thresh)
            if group_method == 'C':     
                self.sort_vids_C(vid_dic=self.video_dic, video=vid, oxygen_thres=self.oxygen_thresh, depth_thresh=self.depth_thresh)
            if group_method == 'D':     
                self.sort_vids_D(vid_dic=self.video_dic, video=vid, oxygen_thres=self.oxygen_thresh, time_thresh1=self.time_thresh1, time_thresh2=self.time_thresh2)
        self.sorted_videos = pd.DataFrame(self.sorted_videos)
        
        # 3) Calculate swimming stats for each video and create a dictionary of processed videos sorted by the chemical group (from step 2)
        self.groups = {}
        for l in self.sorted_videos[0].unique():
            group = []
            for v in range(len(self.sorted_videos)):
                if self.sorted_videos.iloc[v,0] == l:
                    group.append(Video(video=self.video_dic[self.sorted_videos.iloc[v,1]], profile=self.sorted_videos.iloc[v,1], group=l, zoop_class=self.classifier))
                    self.groups[l] = group
        self.group_list = list(self.groups)
        
        # 4) For each group of videos, generate some overall statistics 
        self.all_group_data = []
        for g in self.group_list:
            self.all_group_data.append(Group(group_vids=self.groups[g], group=g))
        
        # 5) Save output file for plotting
        if save:
            self.export_csv_long()
            self.export_metadata()
            self.export_markov()
    
    def sort_vids_A(self, vid_dic=None, video=None, oxygen_thres=None):
        '''Sorts videoes into hypoxic and normoxic -- most basic
        '''
        line = video[9]
        day_tag = ' days'
        day_tag_ind = line.find(day_tag)
        days = line[0:day_tag_ind]
        if abs(int(days)) < 1:                  # exclude videos without a good CTD match
            if (float(video[6])) <= oxygen_thres:
                self.sorted_videos.append(['hypoxic', video[0]])
            else:
                self.sorted_videos.append(['normoxic', video[0]])
    
    def sort_vids_B(self, vid_dic=None, video=None, oxygen_thres=None, time_thresh1=None, time_thresh2=None, depth_thresh=None):
        '''Sorts videos into 8 bins : hypoxic/normoxic, deep/shallow, AM/PM
        '''
        line = video[9]
        day_tag = ' days'
        day_tag_ind = line.find(day_tag)
        days = line[0:day_tag_ind]
        if abs(int(days)) < 1:                  # exclude videos without a good CTD match
            if (float(video[6])) <= oxygen_thres:
                time = datetime.strptime(video[1][:19], "%Y-%m-%d %H:%M:%S")
                if time.hour > time_thresh1 and time.hour < time_thresh2:
                    if (float(video[5])) <= depth_thresh:
                        #print('hypoxic, AM, shallow')
                        self.sorted_videos.append(['hypoxic_AM_shallow', video[0]])
                    else:
                        #print('hypoxic, AM, deep')
                        self.sorted_videos.append(['hypoxic_AM_deep', video[0]])
                else: 
                    if (float(video[5])) <= depth_thresh:
                        #print('hypoxic, PM, shallow')
                        self.sorted_videos.append(['hypoxic_PM_shallow', video[0]])
                    else:
                        #print('hypoxic, PM, deep')
                        self.sorted_videos.append(['hypoxic_PM_deep', video[0]])
            else: 
                time = datetime.strptime(video[1][:19], "%Y-%m-%d %H:%M:%S")
                if time.hour > time_thresh1 and time.hour < time_thresh2:
                    if (float(video[5])) <= depth_thresh:
                        #print('normoxic, AM, shallow')
                        self.sorted_videos.append(['normoxic_AM_shallow', video[0]])
                    else:
                        #print('normoxic, AM, deep')
                        self.sorted_videos.append(['normoxic_AM_deep', video[0]])
                else: 
                    if (float(video[5])) <= depth_thresh:
                        #print('normoxic, PM, shallow')
                        self.sorted_videos.append(['normoxic_PM_shallow', video[0]])
                    else:
                        #print('normoxic, PM, deep')
                        self.sorted_videos.append(['normoxic_PM_deep', video[0]])
    
    def sort_vids_C(self, vid_dic=None, video=None, oxygen_thres=None, depth_thresh=None):
        '''Sorts videos into 4 bins : hypoxic/normoxic, deep/shallow
        '''
        line = video[9]
        day_tag = ' days'
        day_tag_ind = line.find(day_tag)
        days = line[0:day_tag_ind]
        if abs(int(days)) < 1:                  # exclude videos without a good CTD match
            if (float(video[6])) <= oxygen_thres:
                if (float(video[5])) <= depth_thresh:
                    self.sorted_videos.append(['hypoxic_shallow', video[0]])
                else:
                    self.sorted_videos.append(['hypoxic_deep', video[0]])
            else: 
                if (float(video[5])) <= depth_thresh:
                    self.sorted_videos.append(['normoxic_shallow', video[0]])
                else:
                    self.sorted_videos.append(['normoxic_deep', video[0]])
    
    def sort_vids_D(self, vid_dic=None, video=None, oxygen_thres=None, time_thresh1=None, time_thresh2=None):
        '''Sorts videos into 4 bins : hypoxic/normoxic, AM/PM
        '''
        line = video[9]
        day_tag = ' days'
        day_tag_ind = line.find(day_tag)
        days = line[0:day_tag_ind]
        if abs(int(days)) < 1:                  # exclude videos without a good CTD match
            if (float(video[6])) <= oxygen_thres:
                time = datetime.strptime(video[1][:19], "%Y-%m-%d %H:%M:%S")
                if time.hour > time_thresh1 and time.hour < time_thresh2:
                    self.sorted_videos.append(['hypoxic_AM', video[0]])
                else: 
                    self.sorted_videos.append(['hypoxic_PM', video[0]])
            else: 
                time = datetime.strptime(video[1][:19], "%Y-%m-%d %H:%M:%S")
                if time.hour > time_thresh1 and time.hour < time_thresh2:
                    self.sorted_videos.append(['normoxic_AM', video[0]])
                else: 
                    self.sorted_videos.append(['normoxic_PM', video[0]])
    
    def export_csv(self):
        self.df = pd.DataFrame(columns = ['Group', 'Videos', 'Paths', 'Frames', 'Total Jumps', 'Median Paths per Vid', 'Videos with Jumps', 'Avg Jumps per Path', 'Avg Jumps per Frame', 'Avg Cruise Speed', 'Vid w Jumps/Vid']) 
        
        for group in self.all_group_data:
            self.df = self.df.append({'Group' : group.group,
                                    'Videos' : group.total_vids, 
                                    'Paths' : group.total_paths, 
                                    'Frames' : group.total_frames, 
                                    'Total Jumps' : group.group_jumps,
                                    'Median Paths per Vid' : group.med_paths_per_vid,
                                    'Videos with Jumps' : group.vids_w_jumps,
                                    'Avg Jumps per Path': group.overall_avg_jumps_per_path,
                                    'Avg Jumps per Frame': group.group_avg_jump_per_frame,
                                    'Avg Cruise Speed' : group.group_avg_cruise_speed,
                                    'Vid w Jumps/Vid' : group.frac_jumps_vids}, ignore_index = True)
            
        self.df.to_csv(os.path.join(self.rootdir,self.output_file), index=False, sep=',')
        print('Saved output file')
        
    def export_csv_long(self):
        self.df_long = pd.DataFrame(columns = ['group_id', 'vid_id', 'vid_length', 'date_time', 'depth', 'oxygen', 'temp', 'path_id', 'path_length', 'avg_area', 'max_area', 'avg_height', 'max_height', 'avg_width', 'max_width', 
            'avg_cruise_speed', 'cruise_frames', 'avg_jump_speed', 'num_jumps', 'avg_drift_speed', 'drift_frames', 'trans_drift_cruise', 'trans_drift_jump', 'trans_cruise_drift', 'trans_cruise_jump', 'trans_jump_drift', 'trans_jump_cruise',
            'CC_prob', 'CD_prob', 'CJ_prob', 'DC_prob', 'DD_prob', 'DJ_prob', 'JC_prob', 'JD_prob', 'JJ_prob'])
        
        for group in self.all_group_data:
            for video in group.group_vids:
                for path in video.paths_of_interest:
                    self.df_long = self.df_long.append({
                        'group_id': group.group, 
                        'vid_id': video.profile, 
                        'vid_length': self.lookup_table[self.lookup_table[:,0] == video.profile,3][0],
                        'date_time': self.lookup_table[self.lookup_table[:,0] == video.profile,1][0], 
                        'depth': self.lookup_table[self.lookup_table[:,0] == video.profile,5][0], 
                        'oxygen': self.lookup_table[self.lookup_table[:,0] == video.profile,6][0],
                        'temp': self.lookup_table[self.lookup_table[:,0] == video.profile,7][0],
                        'path_id': path.path_ID, 
                        'path_length': path.path_length, 
                        'avg_area': path.path_avg_area,
                        'max_area': path.path_max_area,
                        'avg_height': path.path_avg_height, 
                        'max_height': path.path_max_height, 
                        'avg_width': path.path_avg_width, 
                        'max_width': path.path_max_width, 
                        'avg_cruise_speed': path.path_avg_cruise_speed, 
                        'cruise_frames': path.cruise_frames,
                        'avg_jump_speed': path.path_avg_jump_speed,
                        'num_jumps': path.path_jumps, 
                        'avg_drift_speed': path.path_avg_drift_speed,
                        'drift_frames': path.drift_frames,
                        'trans_drift_cruise': path.trans_drift_cruise,
                        'trans_drift_jump': path.trans_drift_jump,
                        'trans_cruise_drift': path.trans_cruise_drift,
                        'trans_cruise_jump': path.trans_cruise_jump,
                        'trans_jump_drift': path.trans_jump_drift,
                        'trans_jump_cruise': path.trans_jump_cruise,
                        'CC_prob': path.cc_prob,
                        'CD_prob': path.cd_prob,
                        'CJ_prob': path.cj_prob,
                        'DC_prob': path.dc_prob,
                        'DD_prob': path.dd_prob,
                        'DJ_prob': path.dj_prob,
                        'JC_prob': path.jc_prob,
                        'JD_prob': path.jd_prob,
                        'JJ_prob': path.jj_prob
                    }, ignore_index = True)
        
        self.df_long.to_csv(os.path.join(self.rootdir,self.output_file), index=False, sep=',')
        print('Saved output file to: ', os.path.join(self.rootdir,self.output_file))
    
    def export_metadata(self):
        """ export a metadata file, each row is a video (vid, group, path length, depth, oxy, temp, total paths, paths of interest)
            I think Ill need something like this for accurate count plots
            Right now videos that dont contain a path of interest are essentially skipped
        """
        self.meta_df = pd.DataFrame(columns = ['vid_id', 'group_id', 'vid_length', 'depth', 'oxygen', 'temp', 'total_paths', 'paths_of_interest', 'frames_of_interest']) 
        
        for group in self.all_group_data:
            for video in group.group_vids:
                self.meta_df = self.meta_df.append({
                    'vid_id': video.profile,
                    'group_id': group.group,
                    'vid_length': self.lookup_table[self.lookup_table[:,0] == video.profile,3][0],
                    'depth': self.lookup_table[self.lookup_table[:,0] == video.profile,5][0], 
                    'oxygen': self.lookup_table[self.lookup_table[:,0] == video.profile,6][0],
                    'temp': self.lookup_table[self.lookup_table[:,0] == video.profile,7][0],
                    'total_paths': video.num_all_paths,
                    'paths_of_interest': len(video.paths_of_interest),
                    'frames_of_interest': video.total_frames
                }, ignore_index = True)
        
        self.meta_df.to_csv(os.path.join(self.rootdir,self.metadata_file), index=False, sep=',')
        print('Saved metadata file to: ', os.path.join(self.rootdir,self.metadata_file))
    
    def export_markov(self):
        """export csv with group markov model matrices. things are a little messy here bc the size classes arent built in yet
        """
        self.markov_df = pd.DataFrame(columns = ['group_id', 'size_bin','n_paths', 'CC', 'CD', 'CJ', 'DC', 'DD', 'DJ', 'JC', 'JD', 'JJ']) 
        
        for group in self.all_group_data:
            # manually creating size classes here cause I dont want to build it in for this 
            bin_name = ['0-1','1-2','2-3','3-4','4-10']
            bin_len = [len(group.group_speed_states1),len(group.group_speed_states2),len(group.group_speed_states3),len(group.group_speed_states4),len(group.group_speed_states10)]
            bin_obj = [group.trans_matrix_prob1, group.trans_matrix_prob2, group.trans_matrix_prob3, group.trans_matrix_prob4, group.trans_matrix_prob10]
            for i in range(len(bin_name)):
                self.markov_df = self.markov_df.append({
                    'group_id': group.group,
                    'size_bin': bin_name[i],
                    'n_paths': bin_len[i],
                    'CC': bin_obj[i][0,0],
                    'CD': bin_obj[i][0,1],
                    'CJ': bin_obj[i][0,2],
                    'DC': bin_obj[i][1,0],
                    'DD': bin_obj[i][1,1],
                    'DJ': bin_obj[i][1,2],
                    'JC': bin_obj[i][2,0],
                    'JD': bin_obj[i][2,1],
                    'JJ': bin_obj[i][2,2]
                }, ignore_index = True)
                
        self.markov_df.to_csv(os.path.join(self.rootdir,self.markov_file), index=False, sep=',')
        print('Saved markov file to: ', os.path.join(self.rootdir,self.markov_file))
