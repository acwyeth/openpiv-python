
# a script to read in pickled video analysis objects using a look up table, unpickle, and organize final data

# ACW

# 15 July 2022

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
        
        # Extract path data from each group
        self.sorted_data = []
        for l in self.sorted_videos[0].unique():
            for v in range(len(self.sorted_videos)):
                if self.sorted_videos.iloc[v,0] == l:
                    self.video_stats(video_paths=self.video_dic[self.sorted_videos.iloc[v,1]].zoop_paths, zoop_class=self.classifier, group=l)
                    #print(self.path_stats)
                    self.sorted_data.append(self.path_stats)
        self.sorted_data = pd.DataFrame(self.sorted_data, columns=['group','classification','paths','jumps','avg_cruise_speed'])
    
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
    
    def most_frequent(self, List):
        occurence_count = Counter(List)
        return occurence_count.most_common(1)[0][0]
    
    def video_stats(self, video_paths=None, zoop_class=None, group=None):
        paths = 0
        for path in video_paths:
            jumps = 0
            cruise_speed = []
            if not np.isnan(path.x_flow_smoothed).any():                # skip paths with broken smoothing (for now)
                if self.most_frequent(path.classification) == zoop_class:     # only grab paths that are mostly IDed as copepods 
                    paths = paths +1
                    for l in range(len(path.frames)):
                        if path.speed[l] > 100:
                            jumps = jumps + 1
                        else:
                            cruise_speed.append(path.speed[l])
            #print(cruise_speed)
            if len(cruise_speed) > 0:
                avg_cruise_speed = statistics.mean(cruise_speed)
            else:
                avg_cruise_speed = 'NaN'
            self.path_stats = [group, zoop_class, paths, jumps, avg_cruise_speed]

# ==================================================================

test = Analysis(rootdir='/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-17 15:47:46.686791', lookup_file='processed_lookup_table.csv', oxygen_thresh=2, time_thresh=12, depth_thresh=50, classifier='Copepod')

test.sorted_data






# ==================================================================
# ==================================================================
# OLD
# ==================================================================

# directories and filenames

rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-17 15:47:46.686791'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-15 09:54:28.008183'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-04 14:11:00.316190'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-03 12:24:34.550180'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-01 18:01:18.368946'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-07-15 13:39:06.936432'
#rootdir = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-07-18 14:27:30.607462'

lookup_file = 'processed_lookup_table.csv'

# ==================================================================

# Read in processed videos and create a dictionary 

lookup_table = np.genfromtxt(os.path.join(rootdir,lookup_file), dtype = str, delimiter=',', skip_header=0)

video_dic = {}

for file in os.listdir(rootdir):
    if file.endswith('.pickle'):
        print(str(file)[0:10])
        pickle_file = open(os.path.join(rootdir,file),"rb")
        print(pickle_file)
        video_dic[str(file)[0:10]] =  pickle.load(pickle_file)
        pickle_file.close()

keys_list = list(video_dic)

# ==================================================================

# define methods 

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

def video_stats(video, zoop_class):
    for path in video:
        paths = 0
        jumps = 0
        cruise_speed = []
        if not np.isnan(path.x_flow_smoothed).any():                # skip paths with broken smoothing (for now)
            if most_frequent(path.classification) == zoop_class:     # only grab paths that are mostly IDed as copepods 
                paths = paths +1
                for l in range(len(path.frames)):
                    if path.speed[l] > 100:
                        jumps = jumps + 1
                    else:
                        cruise_speed.append(path.speed[l])
        #print(cruise_speed)
        if len(cruise_speed) > 0:
            avg_cruise_speed = statistics.mean(cruise_speed)
        else:
            avg_cruise_speed = 'NaN'
        path_stats = [zoop_class, paths, jumps, avg_cruise_speed]
        return path_stats
        
# ==========================================

for video in lookup_table:
    if (float(video[6])) <= 2:
        line = video[1]
        time = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
        if time.hour <= 12:
            if (float(video[5])) <= 50:
                #print(video)
                print('hypoxic, AM, shallow')
                path_stats = video_stats(video_dic[video[0]].zoop_paths, 'Copepod')
                print(path_stats)
            else:
                print('hypoxic, AM, deep')
                path_stats = video_stats(video_dic[video[0]].zoop_paths, 'Copepod')
                print(path_stats)
        else: 
            if (float(video[5])) <= 50:
                print('hypoxic, PM, shallow')
                path_stats = video_stats(video_dic[video[0]].zoop_paths, 'Copepod')
                print(path_stats)
            else:
                print('hypoxic, PM, deep')
                path_stats = video_stats(video_dic[video[0]].zoop_paths, 'Copepod')
                print(path_stats)
    else: 
        line = video[1]
        time = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
        if time.hour <= 12:
            if (float(video[5])) <= 50:
                print('normoxic, AM, shallow')
                path_stats = video_stats(video_dic[video[0]].zoop_paths, 'Copepod')
                print(path_stats)
            else:
                print('normoxic, AM, deep')
                path_stats = video_stats(video_dic[video[0]].zoop_paths, 'Copepod')
                print(path_stats)
        else: 
            if (float(video[5])) <= 50:
                print('normoxic, PM, shallow')
                path_stats = video_stats(video_dic[video[0]].zoop_paths, 'Copepod')
                print(path_stats)
            else:
                print('normoxic, PM, deep')
                path_stats = video_stats(video_dic[video[0]].zoop_paths, 'Copepod')
                print(path_stats)
                
                
# ==================================================================

# Analsis #2
    # Split paths by oxygen, counts jumps, calculate average cruising speed for amps and copes
    # Jump threshold is basically going to be a placeholder for now

# A) sort paths 
hypoxic_videos = 0
normoxic_videos = 0
hypoxic_copepod_paths = []
normoxic_copepod_paths = []
hypoxic_amphipod_paths = []
normoxic_amphipod_paths = []
broken_flowfield = []
hypoxic_copepod_jumps = 0
hypoxic_copepod_cruise_speeds = []
normoxic_copepod_jumps = 0
normoxic_copepod_cruise_speeds = []
hypoxic_amphipod_cruise_speeds = []
normoxic_amphipod_cruise_speeds = []

thresh = 100

for video in lookup_table:
    if (float(video[6])) <= 2:                                         # set oxygen threshold
        hypoxic_videos = hypoxic_videos + 1
        for path in video_dic[video[0]].zoop_paths:
            if not np.isnan(path.x_flow_smoothed).any():                # skip paths with broken smoothing (for now)
                if most_frequent(path.classification) == 'Copepod':     # only grab paths that are mostly IDed as copepods 
                    hypoxic_copepod_paths.append(path)
                    for l in range(len(path.frames)):
                        if path.speed[l] > thresh:
                            hypoxic_copepod_jumps = hypoxic_copepod_jumps + 1
                        else:
                            hypoxic_copepod_cruise_speeds.append(path.speed[l])
                if most_frequent(path.classification) == 'Amphipod':     # only grab paths that are mostly IDed as copepods 
                    hypoxic_amphipod_paths.append(path)
                    for l in range(len(path.frames)):
                        hypoxic_amphipod_cruise_speeds.append(path.speed[l])
            else:
                broken_flowfield.append(video[0])
    else:
        normoxic_videos = normoxic_videos + 1
        for path in video_dic[video[0]].zoop_paths:
            if not np.isnan(path.x_flow_smoothed).any():
                if most_frequent(path.classification) == 'Copepod':     # only grab paths that are mostly IDed as copepods 
                    normoxic_copepod_paths.append(path)
                    for l in range(len(path.frames)):
                        if path.speed[l] > thresh:
                            normoxic_copepod_jumps = normoxic_copepod_jumps + 1
                        else:
                            normoxic_copepod_cruise_speeds.append(path.speed[l])
                if most_frequent(path.classification) == 'Amphipod':     # only grab paths that are mostly IDed as copepods 
                    normoxic_amphipod_paths.append(path)
                    for l in range(len(path.frames)):
                        normoxic_amphipod_cruise_speeds.append(path.speed[l])
            else:
                broken_flowfield.append(video[0])
                
# Output
print('Copepod Stats')
print('Number of hypoxic paths: ', len(hypoxic_copepod_paths))
print('Number of hypoxic jumps: ', hypoxic_copepod_jumps)
print('Average hypoxic cruising speed: ', statistics.mean(hypoxic_copepod_cruise_speeds))
print('------------------')
print('Number of normoxic paths: ', len(normoxic_copepod_paths))
print('Number of normoxic jumps: ', normoxic_copepod_jumps)
print('Average normoxic cruising speed: ', statistics.mean(normoxic_copepod_cruise_speeds))
print('===================')
print('Amphipod Stats')
print('Number of hypoxic paths: ', len(hypoxic_amphipod_paths))
print('Average hypoxic cruising speed: ', statistics.mean(hypoxic_amphipod_cruise_speeds))
print('------------------')
print('Number of normoxic paths: ', len(normoxic_amphipod_paths))
print('Average normoxic cruising speed: ', statistics.mean(normoxic_amphipod_cruise_speeds))


# ==================================================================

# Analysis #1
    # Split paths into hypoxic/normoxic and calculate mean speed across all videos 
    
# Notes:
    # Only looking at paths where majority of IDs is 'Copepod'
    # Skipping paths with flowfield NaNs for now (need to look into this more...)

hypoxic_paths = []
normoxic_paths = []
broken_flowfield = []

for video in lookup_table:
    #print(video)
    if (float(video[6])) <= 2:                                         # set oxygen threshold
        for path in video_dic[video[0]].zoop_paths:
            if not np.isnan(path.x_flow_smoothed).any():                # skip paths with broken smoothing (for now)
                #print(path.x_flow_smoothed)
                if most_frequent(path.classification) == 'Copepod':     # only grab paths that are mostly IDed as copepods 
                    #print(path)
                    hypoxic_paths.append(path)
            else:
                broken_flowfield.append(video[0])
    else:
        for path in video_dic[video[0]].zoop_paths:
            if not np.isnan(path.x_flow_smoothed).any():                # skip paths with broken smoothing (for now)
                if most_frequent(path.classification) == 'Copepod':
                    normoxic_paths.append(path)
            else:
                broken_flowfield.append(video[0])

# Calculate some simple swimming statistics 

hypx_path_avg_speed = []
norm_path_avg_speed = []

for path in hypoxic_paths:
    #print(path.speed)
    path_avg_speed = statistics.mean(path.speed)
    hypx_path_avg_speed.append(path_avg_speed)

hypox_speed_avg = statistics.mean(hypx_path_avg_speed)
hypox_speed_sd = statistics.stdev(hypx_path_avg_speed)

for path in normoxic_paths:
    path_avg_speed = statistics.mean(path.speed)
    norm_path_avg_speed.append(path_avg_speed)

norm_speed_avg = statistics.mean(norm_path_avg_speed)
norm_speed_sd = statistics.stdev(norm_path_avg_speed)


# Output: 
print("Hypoxic: ", hypox_speed_avg, " +/- ", hypox_speed_sd)
print("Normoxic: ", norm_speed_avg, " +/- ", norm_speed_sd)
print("Broken Flowfields: ", np.unique(broken_flowfield))

# ==================================================================

# comparing two analysis objects 

# changed s2n ratio from 1.3 (vid_dic) to 1.1 (vid_dic2) in PIV analysis 

for video in lookup_table[:5]:
    #print(video)
    if (float(video[6])) > 2:
        for path in video_dic[video[0]].zoop_paths[:5]:
            if most_frequent(path.classification) == 'Copepod':     # only grab paths that are mostly IDed as copepods 
                print(path.x_flow_smoothed)
            
for video in lookup_table2[:5]:
    #print(video)
    if (float(video[6])) > 2:
        for path in video_dic2[video[0]].zoop_paths[:5]:
            if most_frequent(path.classification) == 'Copepod':     # only grab paths that are mostly IDed as copepods 
                print(path.x_flow_smoothed)



# ==================================================================

# Misc. code 

video_dic[keys_list[0]].profile
video_dic[keys_list[1]].zoop_paths
video_dic[keys_list[1]].zoop_paths[1].x_motion

# differences btween path.x_flow_smoothed using two different s2n ratios
# lookup_table
[12.32307888 12.32201592 12.32095297 12.31989002 12.31882707 12.31776412 11.15961114]
[-0.31331357 -1.49240253 -0.36733523  0.75773206  7.78722404  7.19996771]
[ 1.49940505  1.04502921  0.59065337  0.13627752 -0.31809832 -0.77247416 -1.22685    -1.68122584
 -2.13560168 -2.58997752 -3.02106561 -3.07954977 -3.13803392 -3.19651808 -3.25500224 -3.31348639
 -3.37197055 -3.4304547  -3.48893886 -3.54742302 -3.54947288 -3.12826554 -2.7070582  -2.28585086
 -1.86464352 -1.44343618 -1.02222884 -0.6010215  -0.17981416  0.24139318  0.60887376 -0.67934896
 -0.29798428 -0.05604424  0.18589579  0.42783583  0.66977587  0.9117159   1.15365594  1.37086798
  1.50771404  1.6445601   1.78140616  1.91825221  2.05509827  2.19194433  2.32879039  2.46563645
  2.6024825   2.59467435  2.23969609  1.88471782  1.52973956  1.1747613   0.81978303  0.46480477
  0.10982651 -0.24515176  3.39339039  4.16274728  5.21885539  6.27496349  7.3310716   8.3871797 ]
[ 0.06680139  0.08775741  0.071711    0.05566459  0.03961818  0.02357177 -1.32247745 -1.47648914
 -1.63050082 -1.78451251 -1.9385242  -1.87430145 -1.61609253 -1.3578836  -1.09967468 -0.84146575
 -0.58325683 -0.3250479  -0.06683898  0.19136995  0.44957887  0.57451046  0.60614792  0.63778538
  0.66942283  0.70106029  0.73269775  0.76433521  0.79597266  0.31272063  0.35455774  0.33506155
  0.28211084  0.22916012  0.17620941  0.1232587   0.07030798  0.01735727 -0.03559344 -0.08854416
 -0.14149487 -0.16827329 -0.18414659 -0.2000199  -0.2158932  -0.2317665  -0.2476398  -0.2635131
 -0.2793864  -0.2952597   0.41982302  0.67999867  0.92978511  1.17957156  1.429358    1.67914444
  1.92893088  2.17871732  2.42850377  2.67829021  2.92807665  2.72103086  2.41609246  2.11115405
  1.80621565  0.85592449  0.61554671  0.37516894  0.13479116 -0.10558662 -0.34596439 -0.57617711
 -0.80503448 -1.03389185 -1.26274923 -1.4916066  -1.72046398  0.19976463 -0.06618385 -0.33213232
 -0.5980808  -0.35783824 -0.08595874 -0.56177502  0.35975596  1.28128694  2.20281792  3.1243489
  4.04587988  4.96741086  5.88894184  4.47289965  3.05685746  1.64081526  0.22477307 -1.19126912
 -2.60731131 -4.0233535  -5.43939569 -0.27035915 -0.89991147 -1.01603443 -1.13215739 -1.24828036
 -1.36440332 -1.48052628 -1.59664924 -1.7127722  -1.82889516 -1.94501812 -2.01551461 -1.74381255
 -1.47211049 -1.20040842 -0.92870636 -0.6570043  -0.38530224  0.46236517  0.36711031  0.27185545
  0.16453634  0.00091746 -0.16270143 -0.32632031 -0.4899392  -0.65355809 -0.81717697 -0.98079586
 -1.14441475 -1.30803363 -1.42198156 -1.37449887 -1.32701618 -1.27953349 -1.2320508  -1.18456811]
[-0.8892018  -1.20392337 -1.51864493 -1.83336649 -2.14808805 -2.46280961 -2.39071519 -1.97478432
 -1.55885345 -1.14292259 -0.72699172 -0.31106086  0.10487001  0.52080088  0.93673174  1.35266261
  1.45260997  1.24919524  1.5178185   1.78644175  2.055065    2.32368826  2.59231151  2.86093476
  3.12955802  3.39818127  3.22769306  2.8176895   2.40768594  1.99768237  1.58767881  1.17767525
  0.76767169  0.35766813 -0.05233543 -0.46233899 -0.53990097 -0.47894564 -0.4179903  -0.35703497
 -0.29607963 -0.2351243  -0.17416896 -0.11321363 -0.05225829  0.00869705  2.28577113  2.21113645
  2.3647485   2.51836056  2.67197262  2.82558467  2.97919673  3.13280879  3.28642084  3.4400329
  3.50963291  4.46247087  4.02505061  3.58763035  3.15021009  2.71278983  2.27536957  1.83794932
  1.40052906  0.9631088   0.68279561  0.42343003  0.16406445 -0.09530112 -0.3546667 ]

# from lookup_table2
[12.94406023 12.85837492 12.77268962 12.68700431 12.60131901 12.51563371 11.1163578 ]
[ 1.84198961 -0.34912946 -0.0955736   0.15798225  7.34939408  6.25318375]
[ 1.3215841   0.91101942  0.50045474  0.08989006 -0.32067461 -0.73123929 -1.14180397 -1.55236865
 -1.96293332 -2.373498   -2.7546146  -2.66456192 -2.57450923 -2.48445655 -2.39440387 -2.30435118
 -2.2142985  -2.12424582 -2.03419314 -1.94414045 -1.83331292 -1.56667404 -1.30003515 -1.03339626
 -0.76675737 -0.50011849 -0.2334796   0.03315929  0.29979817  0.56643706  0.78837479 -0.6985903
 -0.37120305 -0.20255109 -0.03389912  0.13475284  0.30340481  0.47205677  0.64070873  0.77286701
  0.78642081  0.79997461  0.81352841  0.8270822   0.840636    0.8541898   0.8677436   0.8812974
  0.8948512   0.92978985  1.01605214  1.10231443  1.18857673  1.27483902  1.36110132  1.44736361
  1.5336259   1.6198882   3.33082416  4.10477233  5.192326    6.27987966  7.36743332  8.45498698]
[ 0.06075377  0.07847468  0.06209507  0.04571546  0.02933586  0.01295625 -1.79842678 -2.08594666
 -2.37346654 -2.66098642 -2.9485063  -2.67667349 -1.9076383  -1.1386031  -0.36956791  0.39946729
  1.16850248  1.93753768  2.70657287  3.47560807  4.24464326  3.48128482  1.64525084 -0.19078314
 -2.02681712 -3.8628511  -5.69888509 -7.53491907 -9.37095305 -0.00772593 -0.01001477 -0.00979904
 -0.00821718 -0.00663532 -0.00505347 -0.00347161 -0.00188975 -0.00030789  0.00127396  0.00285582
  0.00443768  0.00227728 -0.00144239 -0.00516206 -0.00888173 -0.01260139 -0.01632106 -0.02004073
 -0.0237604  -0.02748007 -0.01978666  0.28346434  0.6088255   0.93418666  1.25954781  1.58490897
  1.91027013  2.23563129  2.56099244  2.8863536   3.21171476  2.98804046  2.64671572  2.30539097
  1.96406622  0.16018609  0.08563132  0.01107655 -0.06347822 -0.13803299 -0.21258776 -0.31221543
 -0.41518616 -0.51815689 -0.62112762 -0.72409834 -0.82706907 -0.27119029 -0.34871226 -0.42623423
 -0.5037562  -0.26109609  0.0015754  -0.88898928 -0.4401671   0.00865509  0.45747727  0.90629945
  1.35512164  1.80394382  2.25276601  1.32625454  0.39974307 -0.5267684  -1.45327987 -2.37979134
 -3.30630281 -4.23281428 -5.15932575  0.45307797  0.31521794  0.03994109 -0.23533576 -0.51061261
 -0.78588946 -1.06116631 -1.33644316 -1.61172001 -1.88699686 -2.16227371 -2.36605211 -2.03359208
 -1.70113205 -1.36867201 -1.03621198 -0.70375195 -0.37129192  0.5359959   0.50212203  0.46824816
  0.39441812  0.13412601 -0.1261661  -0.38645821 -0.64675032 -0.90704244 -1.16733455 -1.42762666
 -1.68791877 -1.94821088 -2.13166289 -2.06538456 -1.99910624 -1.93282791 -1.86654959 -1.80027126]
[-0.90034141 -1.21389001 -1.52743861 -1.84098721 -2.1545358  -2.4680844  -2.39331512 -1.97337438
 -1.55343364 -1.1334929  -0.71355216 -0.29361142  0.12632932  0.54627006  0.9662108   1.38615153
  1.48630975  1.27669821  1.58966149  1.90262477  2.21558805  2.52855133  2.84151461  3.15447789
  3.46744117  3.78040445  3.59619993  3.14081298  2.68542602  2.23003907  1.77465212  1.31926516
  0.86387821  0.40849125 -0.0468957  -0.50228265 -0.59836788 -0.54474407 -0.49112025 -0.43749643
 -0.38387261 -0.33024879 -0.27662497 -0.22300115 -0.16937733 -0.11575351  1.84379051  2.02415314
  2.21851148  2.41286982  2.60722817  2.80158651  2.99594486  3.1903032   3.38466154  3.57901989
  3.62116271  4.59341148  3.98216592  3.37092037  2.75967482  2.14842926  1.53718371  0.92593815
  0.3146926  -0.29655296 -0.51720752 -0.68578329 -0.85435906 -1.02293483 -1.1915106 ]