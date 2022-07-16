# Script to match each video profile with the nearest CTD cast (in time) and then pull chemical data from nearest depth 
# Reworking script:
    # CTD infastructure stays the same, feed it a single video profile, ouputs are the chemistry for that video 

# I wrote a version of this script in R using the .mat files, but we want to rewrite using the raw .DGC files 

# Notes:    .mat files have a time offset from processing error
#           .DGC files oxygen might not be aligned yet
#           .DGC files timestamp needs to be propogated down for each row
#           I might want to bin .DGC data in 1-m depth bins and average chemistry readings?
#           The CTD is 243mm above the ZooCam
#           Calculate parking depth from the first ~100 readings from each .DGC file (this is currently done in parking_depth.py)
#           Find closest PREVIOUS cast (if the parking depth was changed after a vid was recorded, you want to match video with the old CTD parking depth)

# Notes:
# Oxygen sensor needs times to warm up -- cant average first 100 rows because oxygen is too low
# There is a least one .DGC file that is a really short, stationary readings (breaks code because it doesnt have enough rows)
    # I made a new folder with DGC files only from 2018, also deleted the short files (for now at least)
# When I compare the oxygen (after equilibrated) to the parked depth at the beginning to the oxygen at the same depth during the downcast there is a offset 
    # Is the profilers mixing the water as it moved down? or has the depth offset not been accounted for in the .DCG files
    # It looks like the oxygen value that would match the header is 3 meters lower

# TO DO:
#   ADD CTD/CAM OFFSET!!! 
#       .243 m
#       Need to think about how I want to do this now that I am using header chem data 

# =============================================================================================================================================================

import os
import numpy as np
import glob
import pandas as pd
import csv
import datetime
import time
import sys
import json
import statistics
from statistics import mean
import matplotlib.pyplot as plt
from datetime import datetime
from pytz import timezone

# =============================================================================================================================================================

class CTD():
    """A class to contain and analyze data from CTD downcasts (.DGC files)
    """
    def __init__(self,CTDdir=None,CTDfile=None,CTDload=True,CTDsmpl_int=0.25):
        self.CTDdir=CTDdir
        self.CTDfile=CTDfile
        self.CTDload=CTDload
        # Sample interval in CTD data; apparently 0.25s though not stated in files
        self.CTDsmpl_int=CTDsmpl_int  
        if self.CTDload:
            self.import_CTDfile()
            
    def import_CTDfile(self, CTDdir=None, CTDfile=None, CTDsmpl_int=None, max_header_lines=100, meta_keyword='cast'):
        # Import a raw CTD data file
        if CTDdir is not None:
            self.CTDdir=CTDdir
        if CTDfile is not None:
            self.CTDfile=CTDfile
        if CTDsmpl_int is not None:
            self.CTDsmpl_int=CTDsmpl_int
        self.CTDpath=os.path.join(self.CTDdir,self.CTDfile)
        #print('Loading CTD data from %s' % self.CTDpath)
        self.headernames = ['Scan', 'Pressure', 'Conductivity', 'Temp', 'V0', 'V1', 'V2', 'V3', 'V4','V5', 'O2-V', 'Fluorometer', 'NO3', 'Depth', 'Salinity', 'Sigma-T', 'O2-mg/l','O2-umol/kg', 'O2-sat', 'PAR', 'Turbidity']
        self.ctd_data = pd.read_csv(self.CTDpath, delim_whitespace=True, skiprows=46, header=0, names=self.headernames,na_values='-555')
        
        # extract timestamp from metadata
        with open(self.CTDpath) as csvfile:
            r = csv.reader(csvfile, delimiter=' ')
            for i in range(0, max_header_lines):
                hdr=next(r)
                if hdr[0]==meta_keyword:
                    #print('metadata found in line %d' % i)
                    self.metadata = hdr
                    break
        #print('metadata are :')
        #print(self.metadata)
        
        # Parse data series start time from metadata
        self.datestring = str(self.metadata[4]) + '-' + str(self.metadata[5]) + '-' + str(self.metadata[6]) + \
            ' ' + str(self.metadata[7])
        #print('datastring = ',self.datestring)
        self.ctd_start_date = pd.Timestamp(self.datestring, tz='US/Pacific')
        #print('ctd_start_date = ',self.ctd_start_date)
        self.cast_num = int(self.metadata[3])
        #print('cast_num = ',self.cast_num)

        # Parse parking depths from first readings
        self.parking_depth = mean(self.ctd_data.Depth[0:100])

    def plot_CTDdata(self,start=None,end=None,subplots=False):
        self.ctd_data.plot(subplots=subplots,x="Timestamp",y=["Pressure","Depth"])

    def write_ctd_dataframe(self,save_path=None,save_pandas_format='cvs',overwrite=False):
        # Save the ctd_data dataframe using pandas' built-in methods.
        assert(save_path is not None), "error in write_ctd_dataframe: attemping to save file without a filename..."
        assert(overwrite==True or ~os.path.exists(save_path)),"error in write_ctd_dataframe: file exists and overwrite is False..."
        if 'csv' in save_pandas_format:
            self.ctd_data.to_csv(save_path)
        elif 'json' in save_pandas_format:
            self.ctd_data.to_json(save_path)
        elif ('pickle' in save_pandas_format) or ('pkl' in save_pandas_format):
            self.ctd_data.to_pickle(save_path)

    def read_ctd_dataframe(self,save_path=None,save_pandas_format='csv'):
        # Read a  ctd_data dataframe using pandas' built-in methods
        print('Reading ctd_data from file {} with format {}'.format(save_path,save_pandas_format))
        assert(save_path is not None), "error in read_ctd_dataframe: attemping to read file without a filename..."
        assert(os.path.exists(save_path)),"error in read_ctd_dataframe: file not found..."
        if 'csv' in save_pandas_format:  # compression scheme, if any, is implicit in the file extension
            self.ctd_data=pd.read_csv(save_path,index_col=0,parse_dates=['Timestamp','TimestampADJST','Timestamp_UTC'])
        elif 'json' in save_pandas_format:
            self.ctd_data=pd.read_json(save_path,index_col=0,parse_dates=['Timestamp','TimestampADJST','Timestamp_UTC'])
        elif ('pickle' in save_pandas_format) or ('pkl' in save_pandas_format):
            self.ctd_data=pd.read_pickle(save_path,index_col=0,parse_dates=['Timestamp','TimestampADJST','Timestamp_UTC'])
        # Convert timestamps, using appropriate timezones -- this is ungainly but seems to work
        self.ctd_data['Timestamp_UTC'] = pd.to_datetime(self.ctd_data['Timestamp_UTC'].values,unit='s').tz_localize('UTC')
        self.ctd_data['Timestamp_UTC']=(self.ctd_data['Timestamp_UTC']-pd.Timestamp("1970-01-01",tz='UTC')) / pd.Timedelta('1s')
        self.ctd_data['Timestamp'] = pd.to_datetime(self.ctd_data['Timestamp'].values,unit='s').tz_localize('US/Pacific')
        self.ctd_data['Timestamp']=(self.ctd_data['Timestamp']-pd.Timestamp("1970-01-01",tz='US/Pacific')) / pd.Timedelta('1s')
        self.ctd_data['TimestampADJST'] = pd.to_datetime(self.ctd_data['TimestampADJST'].values,unit='s').tz_localize('US/Pacific')
        self.ctd_data['TimestampADJST']=(self.ctd_data['TimestampADJST']-pd.Timestamp("1970-01-01",tz='US/Pacific')) / pd.Timedelta('1s')

class Analysis():
    '''A class to read in directory of raw CTD data and compile parking depths
    '''
    def __init__(self, CTDdir=None, profile=None):
        self.CTDdir=CTDdir
        self.profile = profile
        self.ctd_files=[]
        self.all_ctd_data=[]
        
        #os.chdir(self.CTDdir)
        fileList = glob.glob(str(self.CTDdir + "/ORCA*_CAST*.DGC"))
        fileList.sort()
        for file in fileList:
            self.ctd_files.append(file)
            #self.ctd = CTD(CTDdir=self.CTDdir, CTDfile=file)
            self.all_ctd_data.append(CTD(CTDdir=self.CTDdir, CTDfile=file))
        print('found {} ctd files'.format(len(self.ctd_files)))

        # extract datenum from video file name
        dt = datetime.utcfromtimestamp(self.profile) 
        self.vid_datnum = dt.strftime("%d-%b-%Y %H:%M:%S")

        # call methods
        self.get_closest_chem()
    
    def get_closest_chem(self):
        # 1) Make a list of all the CDT cast dates (will use this for indexing)
        cast_dates = list(map(lambda val: val.datestring, self.all_ctd_data))

        # 2) Function to return the INDEX of the nearest SMALLER neighbor
        def LowerNeighborIndex(cdt_cast, video):
            index = 0
            while index < len(cdt_cast) and datetime.strptime(video, "%d-%b-%Y %H:%M:%S") > datetime.strptime(cdt_cast[index], "%d-%b-%Y %H:%M:%S"):
                index = index + 1
            #return [(index-1), video, cdt_cast[(index-1)]]
            return (index-1)

        closest_cast_index = LowerNeighborIndex(cast_dates, self.vid_datnum)
                
        self.nearest_earlier_cast = self.all_ctd_data[closest_cast_index].datestring
        #self.time_offset = (self.vid_datnum - datetime.strptime(self.all_ctd_data[closest_cast_index].datestring, "%d-%b-%Y %H:%M:%S")) # gives a timedelta which I dont love but not dealing with now 
        self.temp_avg = mean(self.all_ctd_data[closest_cast_index].ctd_data['Temp'][300:400])
        self.fluor_avg = mean(self.all_ctd_data[closest_cast_index].ctd_data['Fluorometer'][300:400])
        self.depth_avg = mean(self.all_ctd_data[closest_cast_index].ctd_data['Depth'][300:400])
        self.salinity_avg = mean(self.all_ctd_data[closest_cast_index].ctd_data['Salinity'][300:400])
        self.oxygen_mgL_avg = mean(self.all_ctd_data[closest_cast_index].ctd_data['O2-mg/l'][300:400])
