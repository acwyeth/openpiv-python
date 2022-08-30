
# a script to read in pickled video analysis objects using a look up table, unpickle, and organize final data

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
    def __init__(self, path=None):
        self.path_length = len(path.frames)
        self.path_jumps = 0
        self.path_cruise_speeds = []
        self.path_avg_cruise_speed = None
        self.frac_jumps_frames = None
        
        for l in range(self.path_length):
            if path.speed[l] > 100:
                self.path_jumps = self.path_jumps + 1
            else:
                self.path_cruise_speeds.append(path.speed[l])
        
        if len(self.path_cruise_speeds) > 0:
            self.path_avg_cruise_speed = statistics.mean(self.path_cruise_speeds)
        else:
            self.path_avg_cruise_speed = 'NaN'
        
        if self.path_length > 0:
            self.frac_jumps_frames = self.path_jumps / self.path_length

class Video():
    """ A class to organize swimming data on the video level 
    """
    def __init__(self, video=None, group=None, zoop_class=None):
        self.group = group
        self.paths_of_interest = []
        self.total_frames = 0
        self.vid_jumps = 0
        self.jumps_per_path = []
        self.vid_cruise_speed = []
        self.vid_avg_cruise_speed = None
        self.paths_w_jumps = 0
        self.frac_jumps_paths = None
        self.avg_jumps_per_path = None
        
        # sort for paths of interest 
        for path in video.zoop_paths:
            # eventually sort for diff sized copepods as well 
            if not np.isnan(path.x_flow_smoothed).any():                        # skip paths with broken smoothing (for now)
                if self.most_frequent(path.classification) == zoop_class:       # only grab paths that are mostly IDed as copepods 
                    self.paths_of_interest.append(Path(path=path))              # ONLY paths that meet these qualifiers will end up in self.paths_of_interest 
        
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
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]

class Group():
    """ A class to organize swimming data on the chemical/physical grouping level
    """
    def __init__(self, group_vids=None, group=None):
        self.group_vids = group_vids
        self.group = group
        self.total_vids = len(self.group_vids)
        self.total_paths = 0
        self.total_frames = 0
        self.group_jumps = 0
        self.group_cruise_speed = []
        self.avg_jumps_per_path = []
        self.overall_avg_jumps_per_path = None
        self.group_avg_cruise_speed = None
        self.vids_w_jumps = 0
        self.frac_jumps_vids = None
        
        if self.total_vids > 0: 
            for v in self.group_vids:
                self.total_paths = self.total_paths + len(v.paths_of_interest)
                self.total_frames = self.total_frames + v.total_frames
                
                self.group_jumps = self.group_jumps + v.vid_jumps                               # count total number of jumps recorded in this group/environment
                
                if v.vid_avg_cruise_speed != 'NaN' and v.vid_avg_cruise_speed != None:
                    self.group_cruise_speed.extend(v.vid_cruise_speed)
                
                if v.vid_jumps > 0:
                    self.vids_w_jumps = self.vids_w_jumps + 1                                   # count the number of videos in group that contain at least one jump
                    self.avg_jumps_per_path.append(v.avg_jumps_per_path)
            
            if len(self.group_cruise_speed) > 0:
                self.group_avg_cruise_speed = statistics.mean(self.group_cruise_speed)          # average cruise speed in the group (from instantaneous velocities)
            else:
                self.group_avg_cruise_speed = 'NaN'
            
            self.frac_jumps_vids = self.vids_w_jumps / self.total_vids                          # fraction of videos that have a jump
            
            if len(self.avg_jumps_per_path) > 0:
                self.overall_avg_jumps_per_path = statistics.mean(self.avg_jumps_per_path)      # overall average (avg of avg) of the number of jumps in each path in group
            else:
                self.overall_avg_jumps_per_path = 'NaN'

class Analysis():
    """ A class to read in a directory of pickled videos, sort them by designated chemical/physical conditions, and store swimming data on the path, video, and group level
    """
    def __init__(self, rootdir=None, lookup_file=None, oxygen_thresh=None, time_thresh=None, depth_thresh=None, classifier=None, save=True, output_file=None):
        self.rootdir = rootdir
        self.lookup_table = np.genfromtxt(os.path.join(self.rootdir,lookup_file), dtype = str, delimiter=',', skip_header=0)
        self.video_dic = {}
        self.oxygen_thresh = oxygen_thresh
        self.time_thresh = time_thresh
        self.depth_thresh = depth_thresh
        self.classifier = classifier
        self.output_file = output_file
        
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
            self.sort_vids(vid_dic=self.video_dic, video=vid, oxygen_thres=self.oxygen_thresh, time_thresh=self.time_thresh, depth_thresh=self.depth_thresh)
        self.sorted_videos = pd.DataFrame(self.sorted_videos)
        
        # 3) Calculate swimming stats for each video and create a dictionary of processed videos sorted by the chemical group (from step 2)
        self.groups = {}
        for l in self.sorted_videos[0].unique():
            group = []
            for v in range(len(self.sorted_videos)):
                if self.sorted_videos.iloc[v,0] == l:
                    group.append(Video(video=self.video_dic[self.sorted_videos.iloc[v,1]], group=l, zoop_class=self.classifier))
                    self.groups[l] = group
        self.group_list = list(self.groups)
        
        # 4) For each group of videos, generate some overall statistics 
        self.all_group_data = []
        for g in self.group_list:
            self.all_group_data.append(Group(group_vids=self.groups[g], group=g))
        
        # 5) Save output file for plotting
        if save:
            self.export_csv()
    
    def sort_vids(self, vid_dic=None, video=None, oxygen_thres=None, time_thresh=None, depth_thresh=None):
        if (float(video[6])) <= oxygen_thres:
            time = datetime.strptime(video[1][:19], "%Y-%m-%d %H:%M:%S")
            if time.hour <= time_thresh:
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
            line = video[1]
            time = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
            if time.hour <= time_thresh:
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
    
    def export_csv(self):
        self.df = pd.DataFrame(columns = ['Group', 'Videos', 'Paths', 'Frames', 'Total Jumps', 'Videos with Jumps', 'Avg Jumps per Path', 'Avg Cruise Speed', 'Vid w Jumps/Vid']) 
        
        for group in self.all_group_data:
            self.df = self.df.append({'Group' : group.group,
                                    'Videos' : group.total_vids, 
                                    'Paths' : group.total_paths, 
                                    'Frames' : group.total_frames, 
                                    'Total Jumps' : group.group_jumps,
                                    'Videos with Jumps' : group.vids_w_jumps,
                                    'Avg Jumps per Path': group.overall_avg_jumps_per_path,
                                    'Avg Cruise Speed' : group.group_avg_cruise_speed,
                                    'Vid w Jumps/Vid' : group.frac_jumps_vids}, ignore_index = True)
            
        self.df.to_csv(os.path.join(self.rootdir,self.output_file), index=False, sep=',')
        print('Saved output file')


# ==================================================================

test = Analysis(rootdir='/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-17 15:47:46.686791', lookup_file='processed_lookup_table.csv',
    oxygen_thresh=2, time_thresh=12, depth_thresh=50, classifier='Copepod', save=True, output_file='post_processed_swimming_data.csv')