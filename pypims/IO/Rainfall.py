#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rainall
To do:
    To read, compute, and show rainfall data
-----------------    
Created on Tue Jun 23 21:32:54 2020

@author: ming
"""
import time
import numpy as np
from scipy import stats
from datetime import datetime
from datetime import timedelta
from .Raster import Raster
from . import indep_functions as indep_f
class Rainfall:
    """ a class to set rainfall data for hipims
    ---------
    Essential attrs:
    time_s: 1-col array, time in seconds
    mask_header: dictionary showing mask georeference
    mask_dict: dict with two keys:'value' and 'index', providing int array
               showing rain source number and their index respectively
    rain_rate: numpy array m/s
    attrs: summary of the object
    ----------
    Optional attrs:
    subs_in: tuple of row and col number, provide subs of the mask array values
           inside the model domain. Only available when dem_ras is given
    start_date: start date and time of time zero in time_s
    time_dt: datetime format of time_s
    ----------
    Methods:
        set_mask
        set_source
        set_start_date
        get_time_series
        get_spatial_map
        get_valid_rain_rate
        get_attrs
    """
    def __init__(self, rain_mask, rain_source, source_sep=',', dem_ras=None):
        """initialize rainfall object with source file and mask file
        rain_mask: str [filename of a Raster endswith .gz/asc/tif]
                   numpy int array with the same shape with DEM array
                   a Raster object.
                   if rain_mask is a scalar or array, dem_ras must be provided.
        rain_source: numpy array the 1st column is time in seconds, 2nd to
             the end columns are rainfall rates in m/s.
                     str [filename of a csv file for rainfall source data]
        dem_ras:  a Raster object for DEM
        source_sep: delimeter of the rain source file
        """
        self.set_mask(rain_mask, dem_ras)
        self.set_source(rain_source, source_sep)
        del rain_mask
        
    def set_mask(self, rain_mask, dem_ras=None):
        """Set rainfall mask from a scalar or a grid (object/file)
        if rain_mask is a scalar or array, dem_ras must be provided
        """
        if type(rain_mask) is str:
            obj_rain_mask = Raster(rain_mask)
            print(rain_mask+ ' read')
        elif hasattr(rain_mask, 'header'):
            obj_rain_mask = rain_mask
        else: # numpy array or a scalar, dem_ras must be provided
            rain_mask = np.array(rain_mask).astype('int32')
            dem_shape = dem_ras.array.shape
            if rain_mask.size > 1:
                if rain_mask.shape != dem_shape:
                    raise ValueError('The shape of rainfall_mask array '
                             'is not consistent with DEM')
            mask_array = np.zeros(dem_shape, dtype=np.int32)+rain_mask
            obj_rain_mask = Raster(array=mask_array, header=dem_ras.header)
        if hasattr(dem_ras, 'header'):
            # rain mask resample to the same shape with DEM
            self.mask_dict = indep_f._mask2dict(obj_rain_mask, dem_ras.header)
            self.mask_header = dem_ras.header
            self.subs_in = np.where(~np.isnan(dem_ras.array))
        else:
            self.mask_dict = indep_f._mask2dict(obj_rain_mask)
            self.mask_header = obj_rain_mask.header
    
    def set_source(self, rain_source, delimiter=','):
        """Set rainfall mask from a numpy array or a csv file
        """
        if type(rain_source) is str:
            print('reading '+rain_source)
            rain_source = np.loadtxt(rain_source, delimiter=delimiter)
        elif type(rain_source) is np.ndarray:
            pass
        else:
            raise IOError('rain_source must be either a filename or a numpy'
                          ' array')
#        _ = rp._check_rainfall_rate_values(rain_source, times_in_1st_col=True)
        self.time_s = rain_source[:, 0 ]
        self.rain_rate = rain_source[:, 1:]        
        self.get_attrs()

    def get_time_series(self, method='mean', rain_rate_valid=None):
        """ Plot time series of average rainfall rate inside the model domain   
        method: 'mean'|'max','min','mean'method to calculate gridded rainfall 
        over the model domain
        """
        start = time.perf_counter()
        if rain_rate_valid is None:
            rain_rate_valid, _ = self.get_valid_rain_rate() # time-source id
        if method == 'mean':
            value_y = np.mean(rain_rate_valid, axis=1) # m/s
        elif method == 'max':
            value_y = np.max(rain_rate_valid, axis=1) # m/s
        elif method == 'min':
            value_y = np.min(rain_rate_valid, axis=1) # m/s
        elif method == 'median':
            value_y = np.median(rain_rate_valid, axis=1) # m/s
        elif method == 'sum':
            value_y = np.sum(rain_rate_valid, axis=1) # m/s
        else:
            raise ValueError('Cannot recognise the calculation method')
        value_y =  value_y*3600*1000 # mm/h
        time_series = np.c_[self.time_s, value_y]
        end = time.perf_counter()
        print('get_time_series took'+str(end-start))
        return time_series

    def get_spatial_map(self, method='sum'):
        """Get spatial rainfall map over time series
        rain_mask_obj: asc file name or Raster object for rain mask
        cellsize: resample the rain_mask to a new grid (with larger cellsize)
        method: sum|mean caculate method for each cell, sum by time or mean by time 
        """
        # caculate rain source
        time_s = self.time_s
        rain_total = np.trapz(self.rain_rate, time_s, axis=0)*1000 #mm
        unit_str = 'mm/h'
        if method == 'sum':
            cell_rain = rain_total #mm
            unit_str = 'mm'
        elif method == 'mean':
            cell_rain = rain_total/(time_s.max()-time_s.min())*3600 #mm/h
        elif method == 'max':
            cell_rain = np.max(self.rain_rate, axis=0)*1000*3600 #mm/h
        elif method== 'min':
            cell_rain = np.min(self.rain_rate, axis=0)*1000*3600 #mm/h
        elif method== 'median':
            cell_rain = np.median(self.rain_rate, axis=0)*1000*3600 #mm/h
        else:
            raise ValueError('Cannot recognise the calculation method')    
        mask_values = self.mask_dict['value']
        rain_values = cell_rain[mask_values.astype('int32')]
        rain_dict = {'value':rain_values, 'index':self.mask_dict['index']}
        array_shape = (self.mask_header['nrows'], self.mask_header['ncols'])
        rain_array = indep_f._dict2grid(rain_dict, array_shape)
        return rain_array, unit_str
    
    def get_valid_rain_rate(self, unique=True):
        """Get a rain rate array over valid cells
        row is time axis, col is source id axis
        """
        array_shape = (self.mask_header['nrows'], self.mask_header['ncols'])
        mask_array = indep_f._dict2grid(self.mask_dict, array_shape)
        
        if hasattr(self, 'subs_in'):
            mask_valid = mask_array[self.subs_in]
        else:
            mask_valid = mask_array.flatten()
        mask_valid = mask_valid.astype('int32')
        if unique:
            rain_rate_valid = self.rain_rate[:, np.unique(mask_valid)]
        else:
            rain_rate_valid = self.rain_rate[:, mask_valid]
        return rain_rate_valid, mask_valid
    
    def set_start_date(self, datetime_str, fmt):
        """Set start date for the time series
        """
        self.start_date = datetime.strptime(datetime_str, fmt)
        time_delta = np.array([timedelta(seconds=i) for i in self.time_s])
        self.time_dt = self.start_date+time_delta
    
    def get_attrs(self):
        """ define rain_attr
        """
        rain_rate_valid, mask_valid = self.get_valid_rain_rate()
        time_s = self.time_s
        cellsize = self.mask_header['cellsize']
        temporal_res = (time_s.max()-time_s.min())/(time_s.size-1) # seconds
        source_uniq, source_counts = np.unique(mask_valid, return_counts=True)
        num_source = source_uniq.size
        mode_value = stats.mode(source_counts)
        spatial_res = np.sqrt(mode_value[0][0])*cellsize
        cell_mean = np.mean(rain_rate_valid, axis=1)
        # total value along time axis for a cell
        rain_total = np.trapz(cell_mean, x=time_s, axis=0)*1000 # mm
        rain_mean = rain_total/(time_s.max()-time_s.min())*3600 # mm/h
        cell_max = np.max(rain_rate_valid, axis=1)*3600*1000
        attrs = {}
        attrs['num_source'] = num_source
        attrs['max'] = [np.max(cell_max).round(2), 'mm/h']
        attrs['sum'] = [rain_total.round(2), 'mm']
        attrs['average'] = [rain_mean.round(2), 'mm/h']
        attrs['spatial_res'] = [spatial_res.round(), 'm']
        attrs['temporal_res'] = [temporal_res.round(), 's']
        self.attrs = attrs
    
    def get_source_array(self):
        """Return a source array, first column is time_s, then continue with
        rain_rate in m/s
        """
        rain_source = np.c_[self.time_s, self.rain_rate]
        return rain_source
    
    def get_mask_array(self):
        """Return a mask array
        """
        array_shape = (self.mask_header['nrows'], self.mask_header['ncols'])
        mask_array = indep_f._dict2grid(self.mask_dict, array_shape)
        return mask_array
    