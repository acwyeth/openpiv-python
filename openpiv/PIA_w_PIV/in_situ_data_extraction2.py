
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
    def __init__(self, path=None, zoop_class=None):
        self.jumps = 0
        self.cruise_speeds = []
        self.frac_jumps_frames = 0
        
        if not np.isnan(path.x_flow_smoothed).any():                        # skip paths with broken smoothing (for now)
            if self.most_frequent(path.classification) == zoop_class:       # only grab paths that are mostly IDed as copepods 
                for l in range(len(path.frames)):
                    if path.speed[l] > 100:
                        self.jumps = self.jumps + 1
                    else:
                        self.cruise_speeds.append(path.speed[l])
        
        self.path_length = len(path.frames)
        
        if len(self.cruise_speeds) > 0:
            self.avg_cruise_speed = statistics.mean(self.cruise_speeds)
        else:
            self.avg_cruise_speed = 'NaN'
        
        if self.path_length > 0:
            self.frac_jumps_frames = self.jumps / self.path_length
        
    def most_frequent(self, List):
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]

class Video():
    """ A class to organize speed data on the video level 
    """
    def __init__(self, video=None, group=None, zoop_class=None):
        self.group = group
        self.all_paths = []
        self.total_jumps = 0
        self.avg_cruise_speeds = []
        self.paths_w_jumps = 0
        self.frac_jumps_paths = 0
        
        for path in video.zoop_paths:
            self.all_paths.append(Path(path=path, zoop_class=zoop_class))
        
        for p in self.all_paths:
            self.total_jumps = self.total_jumps + p.jumps 
            
            if p.avg_cruise_speed != 'NaN':
                self.avg_cruise_speeds.append(p.avg_cruise_speed)
            if len(self.avg_cruise_speeds) > 0:
                self.vid_avg_cruise_speed = statistics.mean(self.avg_cruise_speeds)
            else:
                self.vid_avg_cruise_speed = 'NaN'
            
            if len(self.all_paths) > 0:
                if p.jumps > 0:
                    self.paths_w_jumps = self.paths_w_jumps + 1
                self.frac_jumps_paths = self.paths_w_jumps / len(self.all_paths)

class Analysis():
    """ A class to read in a directory of pickled videos, sort them by chemical conditions, extract swimming stats, export dataframe
    """
    def __init__(self, rootdir=None, lookup_file=None, oxygen_thresh=None, time_thresh=None, depth_thresh=None, classifier=None):
        
        self.lookup_table = np.genfromtxt(os.path.join(rootdir,lookup_file), dtype = str, delimiter=',', skip_header=0)
        self.video_dic = {}
        self.oxygen_thresh = oxygen_thresh
        self.time_thresh = time_thresh
        self.depth_thresh = depth_thresh
        self.classifier = classifier
        
        # Read in pickled video analyses 
        for file in os.listdir(rootdir):
            if file.endswith('.pickle'):
                print(str(file)[0:10])
                pickle_file = open(os.path.join(rootdir,file),"rb")
                print(pickle_file)
                self.video_dic[str(file)[0:10]] =  pickle.load(pickle_file)
                pickle_file.close()
                
        self.keys_list = list(self.video_dic)
        
        # Sort videos into different groups (chem, depth, etc)
        self.sorted_videos = []
        for vid in self.lookup_table:
            self.sort_vids(vid_dic=self.video_dic, video=vid, oxygen_thres=self.oxygen_thresh, time_thresh=self.time_thresh, depth_thresh=self.depth_thresh)
        self.sorted_videos = pd.DataFrame(self.sorted_videos)
        
        # Generate swimming data for each group
        self.sorted_data = []
        for l in self.sorted_videos[0].unique():
            for v in range(len(self.sorted_videos)):
                if self.sorted_videos.iloc[v,0] == l:
                    self.sorted_data.append(Video(video=self.video_dic[self.sorted_videos.iloc[v,1]], group=l, zoop_class=self.classifier))
    
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

# ==================================================================

test = Analysis(rootdir='/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-17 15:47:46.686791', lookup_file='processed_lookup_table.csv', oxygen_thresh=2, time_thresh=12, depth_thresh=50, classifier='Copepod')

test.sorted_data
test.sorted_data[1].group
test.sorted_data[1].all_paths
test.sorted_data[1].all_paths[1].jumps
test.sorted_data[1].frac_jumps_paths