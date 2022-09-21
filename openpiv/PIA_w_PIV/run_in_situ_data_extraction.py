
# script to run in_situ_data_extraction.py

# ACW 20 Sept 2022

# Notes/To-Dos: 
    # These time thresholds are very wrong

import in_situ_data_extraction2 as ide

# =========================================================================================

#analysis_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-08-17 15:47:46.686791'
analysis_folder = '/home/dg/Wyeth2/IN_SITU_MOTION/analysis_output/2022-09-12 11:22:58.812187'

# =========================================================================================

test = ide.Analysis(rootdir=analysis_folder, lookup_file='processed_lookup_table.csv',
    oxygen_thresh=2, time_thresh=12, depth_thresh=50, classifier='Copepod', save=True, output_file='post_processed_swimming_data.csv')

# =========================================================================================

# all data
test.all_group_data
test.video_dic
test.sorted_videos

# group level
test.all_group_data[0].group
test.all_group_data[0].group_vids
test.all_group_data[0].group_cruise_speed
test.all_group_data[0].group_avg_cruise_speed

# video level
test.all_group_data[0].group_vids[1]
test.all_group_data[0].group_vids[1].paths_of_interest
test.all_group_data[0].group_vids[1].vid_cruise_speed
test.all_group_data[0].group_vids[1].vid_avg_cruise_speed

# path level
test.all_group_data[0].group_vids[1].paths_of_interest[0]
test.all_group_data[0].group_vids[1].paths_of_interest[0].path_length
test.all_group_data[0].group_vids[1].paths_of_interest[0].path_cruise_speeds
test.all_group_data[0].group_vids[1].paths_of_interest[0].path_avg_cruise_speed

