#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Landcover
To do:
    To read, compute, and show Landcover types
-----------------    
Created on Wed Jun 24 09:20:53 2020

@author: ming
"""
import numpy as np
from .Raster import Raster
from . import indep_functions as indep_f
class Landcover:
    """ A class to set landcover data and use it to set grid parameters
    Essential attributes:
        mask_header: dictionary showing mask georeference
        mask_dict: dict with two keys:'value' and 'index', providing int array
               showing landcover type and their index respectively
        
    """
    def __init__(self, ras_data, dem_ras=None):
        """
        """
        if type(ras_data) is str:
            obj_landcover = Raster(ras_data)
        elif hasattr(ras_data, 'header'):
            obj_landcover = ras_data
        if hasattr(dem_ras, 'header'):
            # landcover resample to the same shape with DEM
            self.mask_dict = indep_f._mask2dict(obj_landcover, dem_ras.header)
            self.mask_header = dem_ras.header
            self.subs_in = np.where(~np.isnan(dem_ras.array))
        else:
            self.mask_dict = indep_f._mask2dict(obj_landcover)
            self.mask_header = obj_landcover.header
    
    def get_mask_array(self):
        """Return a mask array
        """
        array_shape = (self.mask_header['nrows'], self.mask_header['ncols'])
        mask_array = indep_f._dict2grid(self.mask_dict, array_shape)
        return mask_array
    
    def to_grid_parameter(self, param_value, land_value, default_value=0):
        """ Set grid parameter according to landcover data
        param_value: scalar or a list of scalar
        land_ids: index representing landcover, scalar, list of scalar,
                  or list of list
        """
        mask_array = self.get_mask_array() #landcover value
        param_array = mask_array*0+default_value
        if type(param_value) is list:
            for i in np.arange(len(param_value)):
                onevalue = param_value[i]
                one_ids = land_value[i]
                ind = np.isin(mask_array, one_ids)
                param_array[ind] = onevalue
        else:
            ind = np.isin(mask_array, land_value)
            param_array[ind] = param_value
        return param_array

