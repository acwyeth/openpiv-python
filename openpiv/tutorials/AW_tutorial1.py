

# ==================================================================

from openpiv import tools, pyprocess, scaling, validation, filters
import numpy as np

import os

# ==================================================================

# Scaling factor to control binning size
fact=3 

# Directory containing frames 
vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1537966443/shrink_piv'

vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1501225077/shrink_piv'

# ==================================================================

# generate a list of frames in given directory
sorted_frames = []
for frame in sorted(os.listdir(vid_dir)):
    #print(frame)
    if frame.endswith(".tif"):
        sorted_frames.append(os.path.join(vid_dir,frame))

# loop over frames and execute PIV analysis
for f in range(len(sorted_frames)):
    #print(f)
    frame_a  = tools.imread(sorted_frames[f])
    frame_b  = tools.imread(sorted_frames[f+1])
    frame_a = (frame_a*1024).astype(np.int32)
    frame_b = (frame_b*1024).astype(np.int32)
    u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        window_size=32*fact, overlap=16*fact, dt=0.02, search_area_size=64*fact, sig2noise_method='peak2peak' )
    print(u,v,sig2noise)
    x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=64*fact, overlap=16*fact )
    u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
    u, v, mask = validation.global_val( u, v, (-1000, 2000), (-1000, 1000) )
    u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )
    tools.save(x, y, u, v, mask, str(vid_dir)+'/test_data_'+str(f)+'.vec' )
    tools.display_vector_field(str(vid_dir)+'/test_data_'+str(f)+'.vec', scale=75, width=0.0035)

# ==================================================================
# OLD CODE:
frame_a  = tools.imread( os.path.join(path,'/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/data/test1/exp1_001_a.bmp'))
frame_b  = tools.imread( os.path.join(path,'/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/data/test1/exp1_001_b.bmp'))

frame_a = (frame_a*1024).astype(np.int32)
frame_b = (frame_b*1024).astype(np.int32)

u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
    window_size=32*fact, overlap=16*fact, dt=0.02, search_area_size=64*fact, sig2noise_method='peak2peak' )

print(u,v,sig2noise)

x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=64*fact, overlap=16*fact )
u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
u, v, mask = validation.global_val( u, v, (-1000, 2000), (-1000, 1000) )
u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )

tools.save(x, y, u, v, mask, '/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/data/test1/test_data.vec' )

tools.display_vector_field('/home/dg/Wyeth2/GIT_repos_insitu/openpiv-python/openpiv/data/test1/test_data.vec', scale=75, width=0.0035)