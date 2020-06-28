#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
init
To do:
    initialize a package
Created on Wed Apr  1 14:56:15 2020

@author: Xiaodong Ming
"""
import numpy as np
from .demo_functions import demo_input
from .demo_functions import demo_output
from .demo_functions import demo_raster
from .demo_functions import get_sample_data
from .InputHipims import InputHipims
from .OutputHipims import OutputHipims
from .indep_functions import load_object, save_as_dict, clean_output, _dict2grid
from .indep_functions import write_times_setup, write_device_setup, write_rain_source
from .Raster import Raster
def load_from_dict(filename):
    """load object from a dictionary and return as an InputHipims object
    """
    obj_dict = load_object(filename)
    
    if 'DEM' in obj_dict:
        dem_dict = obj_dict['DEM']
        obj_dem = Raster(array=dem_dict['array'],
                                header=dem_dict['header'])
        obj_dict.pop('DEM')
    else:
        raise ValueError(filename+' has no key: DEM')
    obj_in = InputHipims(dem_data=obj_dem,
                         num_of_sections=obj_dict['num_of_sections'],
                         case_folder=obj_dict['_case_folder'])
    
    if 'Landcover' in obj_dict:
        ld_dict = obj_dict['Landcover']
        mask_header = ld_dict['mask_header']
        mask_dict = ld_dict['mask_dict']
        array_shape = (mask_header['nrows'], mask_header['ncols'])
        mask_array = _dict2grid(mask_dict, array_shape)
        ras_landcover = Raster(array=mask_array, header=mask_header)
        obj_in.set_landcover(ras_landcover)
        obj_dict.pop('Landcover')
    
    if 'Rainfall' in obj_dict:
        rain_dict = obj_dict['Rainfall']
        mask_header = rain_dict['mask_header']
        mask_dict = rain_dict['mask_dict']
        array_shape = (mask_header['nrows'], mask_header['ncols'])
        mask_array = _dict2grid(mask_dict, array_shape)
        rain_mask = Raster(array=mask_array, header=mask_header)
        rain_source = np.c_[rain_dict['time_s'], rain_dict['rain_rate']]
        obj_in.set_rainfall(rain_mask, rain_source)
        obj_dict.pop('Rainfall')
    
    bound_dict = obj_dict['Boundary']
    for key, value in bound_dict.items():
        obj_in.Boundary.__dict__[key] = value
    obj_dict.pop('Boundary')
    
    summ_dict = obj_dict['Summary']
    for key, value in summ_dict.items():
        obj_in.Summary.__dict__[key] = value
    obj_dict.pop('Summary')
    
    for key, value in obj_dict.items():
        obj_in.__dict__[key] = value
    
    return obj_in