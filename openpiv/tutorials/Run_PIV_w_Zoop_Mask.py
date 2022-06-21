
# A script to batch run PIV_w_Zoop_Mask.py

# ACW June 2022

# To Do:
    # FIXED: when directory doesnt exist
    # FIXED: when directory is empty
    # FIXED: When there is a corrupted .tif file 
        # this is worth double checking as I think about how this data will be integrated with swimming paths 
        
# Notes:
    # x,y,u,v coordinates are currently being transformed to match frames visually -- need to check it matches T3D
    
# Notes:
    # Name figure -- name (number) of first frame 

# ========================================================

import sys
import time 
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
from importlib import reload

sys.path.insert(0, '/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/tutorials')
import PIV_w_Zoop_Mask as piv
import PIV_w_Zoop_Mask_for_PIA as piv2

reload(piv)
reload(piv2)

# ========================================================

# Run a batch of vidoes 

# Parameters 

#dir_list = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests'
dir_list = '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check'
#dir_name = 'shrink_piv'
dir_name = 'shrink'

# pull in other parameters are some point?

# -----------------------------------------

for dir in sorted(os.listdir(dir_list)):
    #plt.close('all')
    path = str(dir_list)+'/'+str(dir)+'/'+str(dir_name)
    print(path)
    try:
        piv.PIV(vid_dir=path, save_setting=False, display_setting=False, verbosity_setting=False)
    except OSError:
        print('ERROR: target folder does not exist')
        continue
    #input("Press Enter to Continue Loop...")

# ========================================================

# Run a single video :

test = piv.PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1535532428/shrink_piv', save_setting=True, display_setting=True, verbosity_setting=True)   # basically no snow 
test = piv.PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1501225077/shrink_piv', save_setting=True, display_setting=True, verbosity_setting=True)  # fast flow
test = piv.PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1502265171/shrink_piv', save_setting=True, display_setting=True, verbosity_setting=True)
test = piv.PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1537007549/shrink_piv', save_setting=True, display_setting=True, verbosity_setting=True)    # very low density snow
test = piv.PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1537852560/shrink_piv', save_setting=True, display_setting=True, verbosity_setting=True)   # basically no snow 
test = piv.PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1535419622/shrink_piv', save_setting=True, display_setting=True, verbosity_setting=True)   # corrupted frames 

test = piv.PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/PIV_test', save_setting=False, display_setting=True, verbosity_setting=False)
#test = piv.PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test', save_setting=False, display_setting=True, verbosity_setting=False)

#test = piv.PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_files_to_check/1535419622/shrink',save_setting=False, display_setting=True, verbosity_setting=False)   # corrupted frame 

plt.close("all")

# ========================================================

# Extra checks  

test.masked_frames[8].ROIlist

# check ROIS
plt.imshow(test.masked_frames[1].frame_image, cmap='gray')
for roi in test.masked_frames[1].ROIlist:
    edge_color='white'                                                                  
    rect = Rectangle((roi.j_beg,roi.i_beg),roi.j_end-roi.j_beg,roi.i_end-roi.i_beg,     
                        linewidth=1,edgecolor=edge_color,facecolor='none')
    plt.gca().add_patch(rect)

# check masked images
#cv2.imshow("masked", test.masked_frames[1].masked_image)       # this line of code is broken for some reason -- crashes python
plt.imshow(test.masked_frames[8].masked_image)

# =========================================================

# Testing ....for_PIA

# inputs = frame1 and frame2, outputs = x,y,u,v,mask

test = piv2.PIV(frame1= '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/SHRINK-8-SPC-UW-1537773780742890-94525972-000500.tif', frame2= '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/SHRINK-8-SPC-UW-1537773780790949-94575976-000501.tif', save_setting=False, display_setting=True, verbosity_setting=True)

plt.imshow(test.masked_frames[1].masked_image)

test.output
test.output[1]