#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created on Wed Apr  1 14:56:15 2020
# Author: Xiaoding Ming

"""

Demo functions
==============

To do:
    * Create the demo

---------------

"""
# import os
import pkg_resources
import numpy as np
from .InputHipims import InputHipims
from .OutputHipims import OutputHipims
from .Raster import Raster

def demo_input(num_of_sections=1, set_example_inputs=True,
               figname=None, dpi=200, **kwargs):
    """ A demonstration to generate a hipims input object
    
    Args:
        set_example_inputs: (True|False) if True, initial condition, boundary 
            condition, rainfall source, and gauge postion will be set to the 
            input object according to sample data.
        figname: (string) if given, a domain map will saved

    """
    dem_file = pkg_resources.resource_filename(__name__,
                                             'sample/DEM.gz')
    obj_in = InputHipims(dem_data=dem_file, num_of_sections=num_of_sections)
    if set_example_inputs:
        __set_defaul_input(obj_in)
    # show model summary print(obj_in)
    obj_in.Summary.display()
    fig, ax = obj_in.domain_show(**kwargs)
    ax.set_title('Domain map')
    if figname is not None:
        fig.savefig(figname, dpi=dpi)
    return obj_in

def demo_output(case_folder, num_of_sections=1):
    """ A demonstration to generate a hipims output object

    Args:
        case_folder: (string) path to case folder
        num_of_sections: (int) number of domains
    
    Note:
        a input folder and files must be created before using this function
    """
    obj_out = OutputHipims(case_folder=case_folder,
                           num_of_sections=num_of_sections)
    return obj_out

def demo_raster(figname=None):
    """ A demonstration to read and show raster files

    Args:
        figname: (string) the file name to save the figure
    """
    dem_file = pkg_resources.resource_filename(__name__,
                                             'sample/Example_DEM.asc')
    obj_ras = Raster(dem_file)
    fig, ax = obj_ras.mapshow(figname=figname, relocate=True, scale_ratio=1000)
    ax.set_title('The Upper Lee catchment DEM (mAOD)')
    return obj_ras

def get_sample_data():
    """ Get sample data for demonstartion

    Returns:
        A DEM raster object and a dictionary with boundary_condition,
        rain_source, and gauges_pos data
    """
    dem_file = pkg_resources.resource_filename(__name__,
                                             'sample/DEM.gz')
    demo_data = {
        'boundary_condition': [
            {'polyPoints': np.array([[1427, 195],
                                 [1446, 243]]),
             'type': 'open',
             'hU': [[0, 30], [60, 20]]},
            {'polyPoints': np.array([[58, 1645],
                                 [72, 1170]]),
             'type': 'open',
             'h': [[0, 10], [60, 10]]}],
        'rain_source': np.array([[0, 2.77777778e-06],
                                 [1800, 2.77777778e-05],
                                 [3600, 0]]),
        'gauges_pos': np.array([[300, 1600],
                                [400, 1200]])
    }
    return dem_file, demo_data
    
# =============private functions==================
def __set_defaul_input(obj_in):
    """Set some default values for an InputHipims object
    """
    # load data for the demo
    _, demo_data = get_sample_data()
    # define initial condition
    h0 = obj_in.DEM.array+0
    h0[np.isnan(h0)] = 0
    h0[h0 < 50] = 0
    h0[h0 >= 50] = 1
    # set initial water depth (h0) and velocity (hU0x, hU0y)
    obj_in.set_initial_condition('h0', h0)
    obj_in.set_initial_condition('hU0x', h0*0.0001)
    obj_in.set_initial_condition('hU0y', h0*0.0002)
    # define boundary condition
    bound_list = demo_data['boundary_condition']
    obj_in.set_boundary_condition(bound_list, outline_boundary='fall')
    # define and set rainfall mask and source (two rainfall sources)
    rain_source = demo_data['rain_source']
    obj_in.set_rainfall(rain_mask=0, rain_source=rain_source)
    # define and set monitor positions
    gauges_pos = demo_data['gauges_pos']
    obj_in.set_gauges_position(gauges_pos)
