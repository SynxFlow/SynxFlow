#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
summary
To do:
    To generate, store, write, and read summary information of a flood model
-----------------    
Created on Fri Jun  5 15:44:18 2020

@author: Xiaodong Ming
"""
#%%
import os
import copy
import numpy as np
import json
from . import rainfall_processing as rp
class Summary:
    """ Summary information of a flood case
    Attributes:
        grid_attr
        model_attr
        boundary_attr
        rain_attr
        params_attr
        initial_attr
    Methods:
        set_grid_attr
        set_model_attr
        set_boundary_attr
        set_rain_attr
        set_params_attr
        set_initial_attr
        display
        
    """
    # default parameters
    grid_attr = {'area':0, # domain area in meter
                 'shape':(1, 1), # tuple of int, grid size rows and cols
                 'cellsize':1,
                 'num_cells':0, # number of valid cells
                 'extent_dict':{'left':0, 'right':0, 'bottom':0, 'top':0},
                 }
    model_attr = {'case_folder':None, # string
                  'birthday':None,
                  'num_GPU':1,
                  'run_time':(0, 3600, 600, 3600), # start, end, write_interval
                  'num_gauges': 0} 
                              
    boundary_attr = {'num_boundary':1,
                     'boundary_details':'outline'}
    rain_attr = {'num_source':0,
                 'average':0,
                 'sum':0,
                 'max':0,
                 'spatial_res':1000, # meter
                 'temporal_res':0 # second
                 }
    params_attr = {'manning':0, 'sewer_sink':0,
                   'cumulative_depth':0, 'hydraulic_conductivity':0,
                   'capillary_head':0, 'water_content_diff':0}
    initial_attr = {'h0':0, 'hU0x':0, 'hU0y':0}
    
    
    def __init__(self, data_in): 
        if type(data_in) is str: # a jason file
            self.__setup_from_file(data_in)
        elif hasattr(data_in, 'DEM'): # a case_obj or dem_obj
            self.__setup_from_object(data_in)

    def __str__(self):
        """
        To show object summary information when it is called in console
        """
        self.display()
        return  self.__class__.__name__
#    __repr__ = __str__
    
    def __setup_from_file(self, filename):
        """Setup object from a file
        """
        summary_dict = load_summary(filename)
        self.grid_attr = summary_dict['grid_attr']
        self.model_attr = summary_dict['model_attr']
        self.boundary_attr = summary_dict['boundary_attr']
        self.rain_attr = summary_dict['rain_attr']
        self.initial_attr = summary_dict['initial_attr']
        self.params_attr = summary_dict['params_attr']
    
    def __setup_from_object(self, obj_in):
        """Setup from a InputHipims object
        """
        
        dem_array = obj_in.DEM.array
        dem_header = obj_in.DEM.header
        extent_dict = obj_in.DEM.extent_dict
        self.set_grid_attr(dem_array, dem_header, extent_dict)

        dt_fmt = '%Y-%m-%d %H:%M'
        self.set_model_attr(case_folder=obj_in.get_case_folder(),
                            birthday=obj_in.birthday.strftime(dt_fmt),
                            num_GPU=obj_in.num_of_sections,
                            run_time=[obj_in.times.tolist(), 's'],
                            num_gauges=obj_in.attributes['gauges_pos'].shape[0]
                            )
        if hasattr(obj_in, 'Rainfall'):
            self.rain_attr = obj_in.Rainfall.attrs
        else:
            rain_source = obj_in.attributes['precipitation_source']
            rain_mask = obj_in.attributes['precipitation_mask']
            rain_mask = rain_mask+dem_array*0
            cellsize = obj_in.DEM.header['cellsize']
            self.set_rain_attr(rain_source, rain_mask, cellsize)
        
        if hasattr(obj_in, 'Boundary'):
            self.set_boundary_attr(obj_in.Boundary)
        
        self.valid_ind = ~np.isnan(dem_array)
        args_dict = copy.copy(self.params_attr)
        for param_name in args_dict.keys():
            args_dict[param_name] = obj_in.attributes[param_name]
        self.set_params_attr(**args_dict)
        self.set_initial_attr(h0=obj_in.attributes['h0'],
                              hU0x=obj_in.attributes['hU0x'],
                              hU0y=obj_in.attributes['hU0y'])

    def set_boundary_attr(self, obj_boundary):
        self.boundary_attr['num_boundary'] = obj_boundary.num_of_bound
        smry_dict = obj_boundary.get_summary()
        self.boundary_attr['boundary_details'] = smry_dict['Boundary details']
        
    def set_params_attr(self, **kw):
        names = self.params_attr.keys()
        for key, value in kw.items():
            if key in names:
                if np.array(value).size==1:
                    self.params_attr[key] = value
                else:
                    values = value[self.valid_ind]
                    values, counts = np.unique(values, return_counts=True)
                    ratios = counts/counts.sum()
                    ratios = ratios.round(4)*100
                    ratios_str = str(ratios.tolist())+'%'
                    self.params_attr[key] = [values.tolist(), ratios_str]
            else:
                print(key+' is not a grid parameter')
    
    def set_initial_attr(self, **kw):
        names = self.initial_attr.keys()
        for key, value in kw.items():
            if key in names:
                if np.array(value).size==1:
                    self.initial_attr[key] = value
                else:
                    values = value[self.valid_ind]
                    ratios = np.sum(values != 0)/values.size
                    self.initial_attr[key] = 'Wet ratio: {:.2f}%'.format(
                            ratios*100)
            else:
                print(key+' is not a grid parameter')
    
    def set_grid_attr(self, dem_array, dem_header, extent_dict):
        """ assign grid_attr
        """
        cellsize = dem_header['cellsize']
        shape = dem_array.shape
        num_cells = ~np.isnan(dem_array)
        num_cells = num_cells.sum()
        area_value = num_cells*cellsize**2
        if area_value>1e7:
            area_value = area_value/1e6
            area_unit = 'km^2'
        else:
            area_unit = 'm^2'
        grid_attr = {'area':[area_value, area_unit],
                     'shape':shape,
                     'cellsize':[cellsize, 'm'],
                     'num_cells':int(num_cells),
                     'extent':extent_dict
                     }
        self.grid_attr = grid_attr  
        
    def set_model_attr(self, **kw):
        """ assign case_folder, num_GPU, run_time, gauge_pos
        """
        for key, value in kw.items():
            if type(value) is np.ndarray:
                value = value.tolist()
            self.model_attr[key] = value
    
    def set_rain_attr(self, rain_source, rain_mask, cellsize):
        """ define rain_attr
        """
        times = rain_source[:,0]
        temporal_res = (times.max()-times.min())/times.size # seconds
        num_cells = rain_mask[~np.isnan(rain_mask)]
        num_source = np.unique(num_cells).size
        spatial_res = np.sqrt(num_cells.size*cellsize/num_source)
        data_sum = rp.get_time_series(rain_source, rain_mask, method='sum')
        rain_total = np.trapz(data_sum[:,1], x=times, axis=0)/1000 # mm
        rain_mean = rain_total/(times.max()-times.min())*3600 # mm/h
        data_max = rp.get_time_series(rain_source, rain_mask, method='max')
        self.rain_attr['num_source'] = num_source
        self.rain_attr['max'] = [np.max(data_max[:,1]).round(2), 'mm/h']
        self.rain_attr['sum'] = [rain_total.round(2), 'mm']
        self.rain_attr['average'] = [rain_mean.round(2), 'mm/h']
        self.rain_attr['spatial_res'] = [spatial_res.round(), 'm']
        self.rain_attr['temporal_res'] = [temporal_res.round(), 's']
    
    def to_dict(self):
        """Convert the object to a dictionary
        """        
        def set_top_dict(origin_dict):
            top_dict = {}    
            for key, value in origin_dict.items():
                if (type(value) is list) or (type(value) is tuple):
                    top_dict[key] = str(value)
                else:
                    top_dict[key] = value
            return top_dict

        summary_dict = {
                'grid_attr':set_top_dict(self.grid_attr),
                'model_attr':set_top_dict(self.model_attr),
                'initial_attr': set_top_dict(self.initial_attr),
                'boundary_attr':set_top_dict(self.boundary_attr),
                'rain_attr':set_top_dict(self.rain_attr),
                'params_attr':set_top_dict(self.params_attr)
                }
        return summary_dict

    def display(self):
        """ display summary information
        """
        def print_dict(one_dict):
            for key, value in one_dict.items():
                print(key,':',value)
        print('---------------------- Grid information ----------------------')
        print_dict(self.grid_attr)
#        print(self.grid_attr)
        print('---------------------- Model information ---------------------')
        print_dict(self.model_attr)
        print('---------------------- Initial condition ---------------------')
        print_dict(self.initial_attr)
        print('---------------------- Boundary condition --------------------')
        print_dict(self.boundary_attr)
        print('---------------------- Rainfall ------------------------------')
        print_dict(self.rain_attr)
        print('---------------------- Parameters ----------------------------')
        print_dict(self.params_attr)
        
    def to_json(self, file_name=None):
        """ write to json file
        """
        summary_dict = self.to_dict()
        if file_name is None:
            case_folder = self.model_attr['case_folder']
            file_name = os.path.join(case_folder, 'readme.txt')
        with open(file_name, 'w') as fp:
            json.dump(summary_dict, fp, indent=4)

    def write_readme(self, filename=None):
        self.to_json(filename)

def load_summary(file_name):
    """ Read a json text file as a dict of Summary
    """
    def js_r(filename):
        with open(filename) as f_in:
            return(json.load(f_in))
    summary_dict = js_r(file_name)
    for key in summary_dict.keys():
        for key_1, value_1 in summary_dict[key].items():
            if type(value_1) is str:
                if value_1[0] == '[' or value_1[0] == '(':
                    summary_dict[key][key_1] = eval(value_1)
    return summary_dict    

def main():
    print('Class to show intput Summary')

if __name__=='__main__':
    main()