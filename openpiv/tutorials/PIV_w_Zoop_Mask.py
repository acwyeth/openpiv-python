

# ==================================================================

from openpiv import tools, pyprocess, scaling, validation, filters
import numpy as np
import os

# everything from PIA
import pickle
import gzip
import copy
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import csv
import time
import random
from matplotlib.patches import Rectangle, Circle, Polygon, Ellipse
import matplotlib.colors as mpl_colors
import matplotlib.patches as mpl_patches
import matplotlib.mathtext as mathtext
import matplotlib.artist as mpl_artist
import matplotlib.image as mpl_image
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.transforms import IdentityTransform

# Specify the backend for matplotlib -- default does not allow interactive mode
from matplotlib import use
use('TkAgg')   # this needs to happen before pyplot is loaded
import matplotlib.pyplot as plt
plt.ion()  # set interactive mode
from matplotlib.backend_bases import MouseButton
from matplotlib import get_backend

#from configGUI import *
#from config23 import *

from tkinter import Tk, simpledialog
from tkinter.filedialog import askopenfilename, asksaveasfilename
#root = Tk()
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
import imageio
global load_dir
load_dir=os.getcwd()
global selector
import pandas as pd

# ==================================================================

# ROI parameters
minThreshold = 20
maxThreshold = 255
min_area = 30
max_area = 5000

# PIV parameters
fact = 3
wndw = 32
ovrlp = 16
srch = 64

# Directory containing frames 
#vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1537966443/shrink_piv'
#vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1501225077/shrink_piv'

# ==================================================================
class ROI():
    """A class to contain and analyze ROIs extracted from ZooCAM frames
    """
    def __init__(self,ROIfile=None,ROIimage=None,counter=None,category=None,label='Unassigned',code=None,bg_color=None,fill_color=None,
                 keypoints=None,i_beg=None,i_end=None,j_beg=None,j_end=None,edge=None,area=None,bbox=None,ellbox=None):
        self.ROIfile=ROIfile
        if ROIimage is not None:
            self.ROIimage=ROIimage
        elif self.ROIfile is not None:
            try:
                self.read_ROI()
            except:
                self.ROIimage=None
                print('ERROR: Failed to load ROI file %s' % self.ROIfile)
        self.counter=counter
        self.edge=edge
        self.area=area
        self.bbox=bbox
        self.ellbox=ellbox
        self.i_beg=i_beg
        self.i_end=i_end
        self.j_beg=j_beg
        self.j_end=j_end
        self.label=label
        self.group=None
             
    def show_image(self,axi):
        axi.imshow(self.ROIimage)
        axi.axis('off')   # turn off axes rendering
        
    def read_ROI(self,ROIfile=None):
            if ROIfile is not None:
                self.ROIfile=ROIfile
            if self.ROIfile is not None:
                try:
                    self.ROIimage=Image.open(self.ROIfile)
                except:
                    self.ROIimage=None
                    print('ERROR: Failed to load ROI file %s' % self.ROIfile)

class ZoopMask():
    """A class to ... mask a single frame?
    """
    def __init__(self, frame_path=None, ROIlist=[], frame_image=None, binary_image=None, contours=[]):
        self.frame_path = frame_path
        self.ROIlist = ROIlist
        
        #call method
        self.masking(frame_path=self.frame_path)
            
    def masking(self, frame_path=None):            
        self.frame_path = frame_path
        self.frame_image=imageio.volread(self.frame_path)   
        
        # Create binary image and fill holes   
        self.binary_image=cv2.threshold(self.frame_image, 20, 225, cv2.THRESH_BINARY)[1]
        # Fill holes:                                                                                         # ACW added -- need graceful way to deal with dropped frames 
        # after example at https://www.programcreek.com/python/example/89425/cv2.floodFill
        frame_floodfill=self.binary_image.copy()
        h, w = self.binary_image.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(frame_floodfill, mask, (0,0), 255);
        frame_floodfill_inv = cv2.bitwise_not(frame_floodfill)
        self.binary_image = self.binary_image.astype(np.uint8) | frame_floodfill_inv.astype(np.uint8)
        #plt.imshow(binary_image, cmap='gray')
        
        # create countours 
        self.contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        nx=len(self.frame_image)
        ny=len(self.frame_image[0])
        
        # Parse ROIs from contours
        self.ROIlist = []
        #ROIpad=5
        ROIpad=1        # black out less
        for ctr in self.contours:
            #print(ctr[:,0,0])
            area=cv2.contourArea(ctr)
            bbox=cv2.boundingRect(ctr)
            if area>min_area and area<max_area:                                        # ACW added to filter ROIs by min area 
                try:
                    ellbox = cv2.fitEllipse(ctr)
                    ell = Ellipse((ellbox[0][0],ellbox[0][1]),ellbox[1][0],ellbox[1][1],angle=ellbox[2],
                                linewidth=1,edgecolor='y',facecolor='none')
                    if display_ROIs:
                        plt.gca().add_patch(ell)
                except:
                    ellbox=None
                i_beg=np.max([np.min(ctr[:,0,1])-ROIpad,0])
                i_end=np.min([np.max(ctr[:,0,1])+ROIpad,nx-1])
                j_beg=np.max([np.min(ctr[:,0,0])-ROIpad,0])
                j_end=np.min([np.max(ctr[:,0,0])+ROIpad,ny-1])
                
                category = 'unknown'
                
                # get blob subimage
                blob_img = Image.fromarray(self.frame_image[i_beg:i_end, j_beg:j_end]) 
                
                self.ROIlist.append(ROI(ROIimage=blob_img,edge=np.squeeze(ctr,axis=1),
                                        area=area,bbox=bbox,ellbox=ellbox,
                                        i_beg=i_beg,i_end=i_end,j_beg=j_beg,j_end=j_end,
                                        category=category))                     
        
        # Black out each ROI
        self.masked_image = cv2.imread(self.frame_path)
        for roi in self.ROIlist:
            # Define an array of endpoints of Hexagon
            points = np.array([[roi.j_beg,roi.i_end],[roi.j_beg,roi.i_beg],[roi.j_end,roi.i_beg],[roi.j_end,roi.i_end]])
            # Use fillPoly() function and give input as image,
            cv2.fillPoly(self.masked_image, pts=[points], color=(0, 0, 0))
        # cv2.imshow("Filled Zoops", self.masked_image)        

class PIV():
    def __init__(self, vid_dir):
        self.vid_dir=vid_dir
        self.sorted_frames = []
        self.masked_frames = []
        
        # generate a list of frames in a given directory
        for frame in sorted(os.listdir(self.vid_dir)):
            if frame.endswith(".tif"):
                self.sorted_frames.append(os.path.join(self.vid_dir,frame))
            
        # fill array of masked frames 
        # Im not really sure what this is going to look like
        for f in range(len(self.sorted_frames)):
            self.masked_frames.append(ZoopMask(frame_path=self.sorted_frames[f]))
        
        # Call method
        self.analysis() 
            
    def analysis(self):
        for m in range(len(self.masked_frames)-1):
            #frame_a  = tools.imread(self.masked_frames[m])
            #frame_b  = tools.imread(self.masked_frames[m+1])
            frame_a = tools.rgb2gray(self.masked_frames[m].masked_image)
            frame_b = tools.rgb2gray(self.masked_frames[m+1].masked_image)            
            frame_a = (frame_a*1024).astype(np.int32)
            frame_b = (frame_b*1024).astype(np.int32)
            
            u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
                window_size=(wndw*fact), overlap=(ovrlp*fact), dt=0.02, search_area_size=(srch*fact), sig2noise_method='peak2peak' )
            print('ROUND 1:')
            print(u, v, sig2noise)
            
            x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=(srch*fact), overlap=(ovrlp*fact) )
            #x, y = pyprocess.get_rect_coordinates( frame_a=frame_a.shape, window_size=(wndw*fact), overlap=(ovrlp*fact) )
            
            u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )   # masks anything with a s2n less than 1.3
            print('ROUND 2:')
            print(u,v,mask)
            
            u, v, mask = validation.global_val( u, v, (-1000, 2000), (-1000, 1000) )    # masks anything outside of these global outliers -- aka any unrealistic flowfield
            print('ROUND 3:')
            print(u,v,mask)
            
            u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
            print('ROUND 4:')
            print(u,v)
            
            #x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 )  # just divides everything by the scaling factor -- probably don't need
            #print('ROUND 5')
            #print(u,v)
            
            tools.save(x, y, u, v, mask, str(self.vid_dir)+'/test_data_'+str(m)+'.vec' )
            #tools.display_vector_field(str(self.vid_dir)+'/test_data_'+str(m)+'.vec', scale=75, width=0.0035)


# ==========================================================================================================

test = PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1501225077/shrink_piv')  # high density -- nans
test = PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1502265171/shrink_piv')
test = PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/PIV_tests/1537007549/shrink_piv')  # lots of nan outputs 

# looks good
test = PIV(vid_dir = '/home/dg/Wyeth2/IN_SITU_MOTION/shrink_tracking_tests/1537773747/motion_test/PIV_test')

# ---------

test.masked_frames[1].ROIlist

# check ROIS
plt.imshow(test.masked_frames[8].frame_image, cmap='gray')
for roi in test.masked_frames[8].ROIlist:
    edge_color='white'                                                                  
    rect = Rectangle((roi.j_beg,roi.i_beg),roi.j_end-roi.j_beg,roi.i_end-roi.i_beg,     
                        linewidth=1,edgecolor=edge_color,facecolor='none')
    plt.gca().add_patch(rect)

# check masked images
cv2.imshow("masked", test.masked_frames[8].masked_image)
