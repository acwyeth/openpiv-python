
# A script to read in a directory of ZooCam video frames (tif images), mask out zooplankton (FillPoly black), and calculate flowfield displacement using PIV machinery

# ACW June 2022

#cd /home/dg/Wyeth2/GIT_repos_insitu/openpiv-python
#python3.8

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
import matplotlib
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
from tkinter import Tk, simpledialog
from tkinter.filedialog import askopenfilename, asksaveasfilename
Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
import imageio
global load_dir
load_dir=os.getcwd()
global selector
import pandas as pd

from importlib import reload
reload(tools)

# ==================================================================

# ROI parameters ------
minThreshold = 20
maxThreshold = 255
min_area = 30
max_area = 5000

# PIV parameters ------
frm_rt = (1/1)     # right now conversation to physical space and time happens at the analysis level!

# NON extended search parameters (srch==window)
# 5 x 4 grid -- currently the best I think 
wndw =  256
ovrlp = 128
srch = 256

# # single arrow -- need to update display code, but otherwise running
# wndw =  650   # 32
# ovrlp = 325  # 16
# srch = 650   # 64

# # 2 x 1 grid
# wndw =  512
# ovrlp = 256
# srch = 512

# # 3 x 2 grid
# wndw =  384
# ovrlp = 192
# srch = 384

# # 5x4 window - extended search (srch>window)
# fact = 3
# wndw =  32
# ovrlp = 16
# srch = 64

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
    """A class to black out zooplankton from a given frame (don't want swimmers included in PIV analysis)
    """
    def __init__(self, frame_path=None, ROIlist=[], frame_image=None, binary_image=None, contours=[]):
        self.frame_path = frame_path
        self.ROIlist = ROIlist
        
        #call method
        try:
            self.masking(frame_path=self.frame_path)
            #print('Masked frame')
        except:
            pass
            #print("ERROR: Could not mask frame") 
    
    def masking(self, frame_path=None):            
        self.frame_path = frame_path                                                                                
        self.frame_image=imageio.volread(self.frame_path)  
        
        # Create binary image and fill holes   
        self.binary_image=cv2.threshold(self.frame_image, 20, 225, cv2.THRESH_BINARY)[1]
        # Fill holes:
        # after example at https://www.programcreek.com/python/example/89425/cv2.floodFill
        frame_floodfill=self.binary_image.copy()
        h, w = self.binary_image.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        cv2.floodFill(frame_floodfill, mask, (0,0), 255);
        frame_floodfill_inv = cv2.bitwise_not(frame_floodfill)
        self.binary_image = self.binary_image.astype(np.uint8) | frame_floodfill_inv.astype(np.uint8)
        #plt.imshow(binary_image, cmap='gray')
        
        # Create countours 
        self.contours, hierarchy = cv2.findContours(self.binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        nx=len(self.frame_image)
        ny=len(self.frame_image[0])
        
        # Parse ROIs from contours
        self.ROIlist = []
        #ROIpad=5
        ROIpad=1        # black out smaller areas
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
        #cv2.imshow("Filled Zoops", self.masked_image)        

class PIV():
    def __init__(self, frame1=None, frame2=None, save_setting=True, display_setting=True, verbosity_setting=True):
        self.frame1 = frame1
        self.frame2 = frame2
        self.frame_pair = [self.frame1, self.frame2]
        self.masked_frames = []
        self.output=None
        
        # Populate array of masked frames 
        for f in range(len(self.frame_pair)):
            self.masked_frames.append(ZoopMask(frame_path=self.frame_pair[f]))
        
        # Call method
        self.analysis(save=save_setting, display=display_setting, verbose=verbosity_setting) 
            
    def analysis(self, save=True, display=True, verbose=True):
        """Method that calls OpenPIV scripts and generates flowfiels between sequetial frames 
        """
        # 1) read in sequential masked frames
        frame_a = tools.rgb2gray(self.masked_frames[0].masked_image)
        frame_b = tools.rgb2gray(self.masked_frames[1].masked_image)            
        frame_a = (frame_a*1024).astype(np.int32)
        frame_b = (frame_b*1024).astype(np.int32)
        
        # 2) divid frames into window, normalize, fft corrlations, corrleation to displacement, signal to noise ratios calculated
        #u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
        #    window_size=(wndw*fact), overlap=(ovrlp*fact), dt=0.02, search_area_size=(srch*fact), sig2noise_method='peak2peak' )
        
        # 2) No longer using the extended search functionality, turned normalization back on to account for this change
        u, v, sig2noise = pyprocess.extended_search_area_piv( frame_a, frame_b, \
            window_size=wndw, overlap=ovrlp, dt=frm_rt, search_area_size=None, sig2noise_method='peak2peak', normalized_correlation=True )
        if verbose == True:
            print('ROUND 1: Raw displacements')
            print(u, v, sig2noise)
        
        # 3) get window coordinates (bins)
        x, y = pyprocess.get_coordinates( image_size=frame_a.shape, search_area_size=srch, overlap=ovrlp )
        
        # 4) mask displacement values with s2n (peak2peak) value below threshold (should think about this thresh more)
        #u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.3 )
        u, v, mask = validation.sig2noise_val( u, v, sig2noise, threshold = 1.1 )       # ACW experiment!
        if verbose == True:
            print('ROUND 2: S2N mask')
            print(u,v,mask)
        
        # Masks anything outside of these global outliers - not applying right now
            # Note to self: If I want to use this fitler -- need to think about reasonable outliers in out setting
        # u, v, mask = validation.global_val( u, v, (-1000, 2000), (-1000, 1000) )
        # print('ROUND 3:')
        # print(u,v,mask)
        
        # TESTING ACW Feb 2 2023
        # Masks anything outside of 2 standard deviations
        u, v, mask = validation.global_std(u, v, std_threshold=1.5)
        if verbose == True:
            print('ROUND 2.5: StDev mask')
            print(u,v,mask)
        
        # 5) Fills in masked grid values using neighboring windows
        u, v = filters.replace_outliers( u, v, method='localmean', max_iter=10, kernel_size=2)
        if verbose == True:
            print('ROUND 3: Filled outliers')
            print(u,v)
        
        # 6) Transform Coordinates 
        # Converts coordinate systems from/to the image based / physical based 
        x, y, u, v = tools.transform_coordinates(x, y, u, v)
        
        # 7) Scales all values -- important for displaying -- not applying right now 
        #x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor = 96.52 ) 
        # if verbose == True:
        #     print('ROUND 4: Scaled (for plotting)')
        #     print(u,v)
        
        # 8) Save output file
        if save == True:
            tools.save(x, y, u, v, mask, str(self.vid_dir)+'/PIV_data_'+str(m)+'.vec' )
            if verbose == True:
                print('File saved to: '+ str(self.vid_dir))
        
        # 9) Display flowfields 
        if display == True:
            #tools.display_vector_field(str(self.vid_dir)+'/PIV_data_'+str(m)+'.vec', scale=75, width=0.0035)
            # doesnt require a file to be saved 
            tools.display_vector_field_AW(x, y, u, v, mask, scale=75, width=0.0035)
        
        # 10) Create flowfield output that will be used in analysis classes
        self.output = np.vstack([m.flatten() for m in [x, y, u, v, mask]])