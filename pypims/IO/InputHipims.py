#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
InputHipims
Generate input files for a hipims flood model
-------------------------------------------------------------------------------
@author: Xiaodong Ming
Created on Tue Mar 31 16:03:57 2020
-------------------------------------------------------------------------------
Assumptions:
- Input DEM is a regular DEM file
- its map unit is meter
- its cellsize is the same in both x and y direction
- its reference position is on the lower left corner of the southwest cell
- All the other grid-based input files must be consistent with the DEM file
To do:
- generate input (including sub-folder mesh and field) and output folders
- generate mesh file (DEM.txt) and field files
- divide model domain into small sections if multiple GPU is used
Structure:
   class InputHipims
       - Initialize an object: __init__
       - set model parameters: 
           set_boundary_condition, set_parameter, set_rainfall, 
           set_gauges_position, set_case_folder, set_runtime, set_device_no,
           add_user_defined_parameter, set_num_of_sections
"""
__author__ = "Xiaodong Ming"
import os
import shutil
import copy
import numpy as np
from datetime import datetime
from .Raster import Raster
from .Boundary import Boundary
from .Rainfall import Rainfall
from .Landcover import Landcover
from .Summary import Summary
from .spatial_analysis import sub2map
from . import indep_functions as indep_f
from . import rainfall_processing as rp
#%% definition of class InputHipims
class InputHipims:
    """To define input files for a HiPIMS flood model case
    Read data, process data, write input files, and save data of a model case.
    Properties (public):
        case_folder: (str) the absolute path of the case folder
        data_folders: (dict) paths for data folders(input, output, mesh, field)
        num_of_sections: (scalar) number of GPUs to run the model
        shape: shape of the DEM array
        header: (dict) header of the DEM grid
        __attributes_default: (dict) default model attribute names and values
        attributes: (dict) model attribute names and values
        times:  (list/numpy array) of four values reprenting model run time in
                seconds: start, end, output interval, backup interval
        device_no: (int) the gpu device id(s) to run model
        param_per_landcover: dict, argument to set grid parameters using Landcover
            object. Keys are grid parameter names. Value is a dict with three
            keys: param_value, land_value, default_value. Refer to Landcover
            for more details.
        -------
        Objects
        -------
        Sections: a list of objects of child-class InputHipimsSub
        Boundary: To create a Boundary object for boundary conditions
            The object contains
                  outline_boundary, a dataframe of boundary type, extent,
                  source data, code, ect..., and a boundary subscrpits
                  tuple (cell_subs).For multi-GPU case, a boundary subscrpits
                  tuple (cell_subs_l) based on sub-grids will be created for 
                  each section
            outline_boundary: (str) 'open'|'rigid', default outline boundary is
                open and both h and hU are set as zero
            boundary_list: (list of dicts), each dict contain keys (polyPoints,
                type, h, hU) to define a IO boundary's position, type, and
                Input-Output (IO) sources timeseries. Keys including:
                1.polyPoints is a numpy array giving X(1st col) and Y(2nd col)
                    coordinates of points to define the position of a boundary.
                    An empty polyPoints means outline boundary.
                2.type: 'open'(flow out flatly), 'rigid'(no outlet),
                        'fall'(water flow out like a fall)
                3.h: a two-col numpy array. The 1st col is time(s). The 2nd col
                     is water depth(m)
                4.hU: a two-col numpy array. The 1st col is time(s). The 2nd
                    col is discharge(m3/s) or a three-col numpy array, the 2nd
                    col and the 3rd col are velocities(m/s) in x and y
                    direction, respectively.
            cell_subs: (list of tuple) subsripts of cells for each boundary
            cell_id: (list of vector) valid id of cells for each boundary
        DEM: a Raster object to provide DEM data [alias: Raster].
        Rainfall: a Rainfall object to provide rainfall data
        Landcover: a Landcover object to provide landcover data for setting
            gridded parameters
            
        Summary: a Summary object to record model information
    Properties (Private):
        _valid_cell_subs: (tuple, int numpy) two numpy array indicating rows
            and cols of valid cells on the DEM array and sorted based on the
            valid_cell_id, starting from the bottom-left valid cell towards
            right and up on the DEM array.
        _outline_cell_subs: (tuple, int numpy) two numpy array indicating rows
            and cols of outline cells on the DEM array and sorted based on the
            valid_cell_id from small to large
    Methods (public):
        set_parameter: set grid-based parameters
        set_boundary_condition: set boundary condition with a boundary list
        set_rainfall: set rainfall mask and sources
        set_gauges_position: set X-Y coordinates of monitoring  gauges
        write_input_files: write all input files or a specific file
        write_grid_files: write grid-based data files
        write_boundary_conditions: write boundary sources, if flow time series
            is given, it will be converted to velocities in x and y directions
        write_gauges_position: write coordinates of monitoring gauges
        write_halo_file: write overlayed cell ID for multiple GPU cases
    Classes:
        InputHipimsSub: child class of InputHiPIMS, provide information
            of each sub-domain
        Boundary: provide information of boundary conditions
        ModelSummary: record basic information of an object of InputHiPIMS
    """
#%%****************************************************************************
#***************************initialize an object*******************************
    # default parameters
    __attributes_default = {'h0':0, 'hU0x':0, 'hU0y':0,
                          'precipitation':0,
                          'precipitation_mask':0,
                          'precipitation_source':np.array([[0, 0], [1, 0]]),
                          'manning':0.035,
                          'sewer_sink':0,
                          'cumulative_depth':0, 'hydraulic_conductivity':0,
                          'capillary_head':0, 'water_content_diff':0,
                          'gauges_pos':np.array([[0, 0], [1, 1]])}
    __grid_files = ['z', 'h', 'hU', 'precipitation_mask',
                  'manning', 'sewer_sink', 'precipitation',
                  'cumulative_depth', 'hydraulic_conductivity',
                  'capillary_head', 'water_content_diff']
    gridded_parameter_keys = ['h0', 'hU0x', 'hU0y', 'manning', 'sewer_sink',
                           'cumulative_depth', 'hydraulic_conductivity',
                           'capillary_head', 'water_content_diff']
    def __init__(self, dem_data=None, num_of_sections=1, case_folder=None,
                 data_path=None):
        """
        dem_data: (Raster object) or (str) provides file name of the DEM data
        data_folder: a path contain at least a DEM file named as 'DEM' with a
            suffix .gz|.asc|.tif. 'landcover' and 'rain_mask' can also be read if
            these files were given with one of the three suffix
        """
        self.attributes = InputHipims.__attributes_default.copy()
        self.num_of_sections = num_of_sections
        self.birthday = datetime.now()
        if case_folder is None:
            case_folder = os.getcwd()
        self._case_folder = case_folder
        if data_path is not None:
            self.data_path = data_path
            self._setup_by_files(data_path)
        else:
            if type(dem_data) is str:
                self.DEM = Raster(dem_data) # create Raster object
            elif type(dem_data) is Raster:
                self.DEM = dem_data
        self.Raster = self.DEM
        self.shape = self.DEM.shape
        self.header = self.DEM.header
        if not hasattr(self, 'Rainfall'):
            self.set_rainfall()
        _ = self.set_runtime()
        # get row and col index of all cells on DEM grid
        self._get_cell_subs()  # add _valid_cell_subs and _outline_cell_subs
        # divide model domain to several sections if it is not a sub section
        # each section contains a "HiPIMS_IO_class.InputHipimsSub" object
        if isinstance(self, InputHipimsSub):
            pass
        else:
            self.__divide_grid()
        self.set_case_folder() # set data_folders
        self.set_device_no() # set the device number
        self.set_boundary_condition(outline_boundary='fall')
        self.set_gauges_position()
        self._initialize_summary_obj()# initialize a Model Summary object
        
    def __str__(self):
        """
        To show object summary information when it is called in console
        """
        self.Summary.display()
        time_str = self.birthday.strftime('%Y-%m-%d %H:%M:%S')
        return  self.__class__.__name__+' object created on '+ time_str
#    __repr__ = __str__
#******************************************************************************
#%%**********************Get object attributes**************************************
    def get_case_folder(self):
        """ Return _case_folder
        """
        return self._case_folder

    def get_data_folders(self):
        """ Return _case_folder
        """
        return self._data_folders

#%%**********************Setup the object**************************************
    def set_boundary_condition(self, boundary_list=None,
                               outline_boundary='fall'):
        """To create a Boundary object for boundary conditions
        """
        if boundary_list is None and hasattr(self, 'Boundary'):
            boundary_list = self.Boundary.boundary_list
        bound_obj = Boundary(boundary_list, outline_boundary)
        valid_subs = self._valid_cell_subs
        outline_subs = self._outline_cell_subs
        if not isinstance(self, InputHipimsSub):
            dem_header = self.header
        # add the subsripts and id of boundary cells on the domain grid
            bound_obj._fetch_boundary_cells(valid_subs,
                                            outline_subs, dem_header)
        self.Boundary = bound_obj
        if hasattr(self, 'Sections'):
            bound_obj._divide_domain(self)       
        if hasattr(self, 'Summary'):
            self.Summary.set_boundary_attr(self.Boundary)

    def set_initial_condition(self, parameter_name, parameter_value):
        """ Set initial condition for h0, hU0x, hU0y
        parameter_name: (str) h0, hU0x, hU0y
        parameter_value: scalar or numpy array with the same size of DEM.
        """
        if parameter_name not in ['h0', 'hU0x', 'hU0y']:
            raise ValueError('Parameter is not recognized: '+parameter_name)
        if type(parameter_value) is np.ndarray:
            if parameter_value.shape != self.shape:
                raise ValueError('The array of the parameter '
                                 'value should have the same '
                                 'shape with the DEM array')
        elif np.isscalar(parameter_value) is False:
            raise ValueError('The parameter value must be either '
                             'a scalar or an numpy array')
        self.attributes[parameter_name] = parameter_value
        self.Summary.set_initial_attr(**{parameter_name:parameter_value})
    
    def set_grid_parameter(self, **kwargs):
        """ Set grid parameter with Landcover object as name=value
        kwargs: Keyword Arguments Specified by a Dictionary
        keyword: name, from grid_parameter_keys
        value:  1. scalar, a uniform parameter value
                2. array, gridded parameter value with the same size of DEM
                3. dict, contain param_value, land_value, default_value=0
        Return: save a parameter dictionary
        """
        if not hasattr(self, 'param_per_landcover'):
            # save arguments to call Landcover.to_grid_parameter()
            self.param_per_landcover = {}
        for keyword, value in kwargs.items():
            if keyword not in self.gridded_parameter_keys:
                raise ValueError('Parameter is not recognized: '+keyword)
            if type(value) is np.ndarray:
                if value.shape != self.shape:
                    raise ValueError('The array of the parameter '+keyword+
                           ' should have the same shape with the DEM array')
                else:
                    self.attributes[keyword] = value
            elif np.isscalar(value):
                self.attributes[keyword] = value
            elif type(value) is dict:
                self.param_per_landcover[keyword] = value
                value_array = self.Landcover.to_grid_parameter(**value)
                value = value_array
            else:
                raise ValueError(keyword+' must be a scalar, array or dict')
            self.Summary.set_params_attr(**{keyword:value})

    def set_rainfall(self, rain_mask=None, rain_source=None):
        """ Set rainfall mask and rainfall source
        rain_mask: str [filename of a Raster endswith .gz/asc/tif]
                   numpy int array with the same shape with DEM array
                   a Raster object
        rain_source: numpy array the 1st column is time in seconds, 2nd to
             the end columns are rainfall rates in m/s.
                     str [filename of a csv file for rainfall source data]
        """
        if not hasattr(self, 'Rainfall'):
            self.Rainfall = Rainfall(self.attributes['precipitation_mask'], 
                                     self.attributes['precipitation_source'],
                                     dem_ras=self.DEM)
        # set rain mask
        if (rain_mask is None) & (rain_source is None): 
            # use default or pre-defined value
            rain_mask = self.attributes['precipitation_mask']
            rain_source = self.attributes['precipitation_source']
            self.Rainfall = Rainfall(rain_mask, rain_source, dem_ras=self.DEM)
        if rain_mask is not None:
            self.Rainfall.set_mask(rain_mask, dem_ras=self.DEM)
        if rain_source is not None:
            self.Rainfall.set_source(rain_source)
        if hasattr(self, 'Summary'):
            self.Summary.rain_attr = self.Rainfall.attrs

    def set_gauges_position(self, gauges_pos=None):
        """Set coordinates of monitoring gauges
        """
        if gauges_pos is None:
            gauges_pos = self.attributes['gauges_pos']
        if type(gauges_pos) is list:
            gauges_pos = np.array(gauges_pos)
        if gauges_pos.shape[1] != 2:
            raise ValueError('The gauges_pos arraymust have two columns')
        self.attributes['gauges_pos'] = gauges_pos
#        self.Summary.add_param_infor('gauges_pos', gauges_pos)
        # for multi_GPU, divide gauges based on the extent of each section
        if hasattr(self, 'Sections'):
            pos_X = gauges_pos[:,0]
            pos_Y = gauges_pos[:,1]
            for obj_section in self.Sections:
                extent = obj_section.DEM.extent
                ind_x = np.logical_and(pos_X >= extent[0], pos_X <= extent[1])
                ind_y = np.logical_and(pos_Y >= extent[2], pos_Y <= extent[3])
                ind = np.where(np.logical_and(ind_x, ind_y))
                ind = ind[0]
                obj_section.attributes['gauges_pos'] = gauges_pos[ind,:]
                obj_section.attributes['gauges_ind'] = ind
        if hasattr(self, 'Summary'):
            self.Summary.set_model_attr(gauges_pos=gauges_pos)

    def set_case_folder(self, new_folder=None, make_dir=False):
        """ Initialize, renew, or create case and data folders
        new_folder: (str) renew case and data folder if it is given
        make_dir: True|False create folders if it is True
        """
        # to change case_folder
        if new_folder is None:
            new_folder = self._case_folder
        self._case_folder = new_folder
        self._data_folders = indep_f._create_io_folders(self._case_folder,
                                               make_dir)
        # for multiple GPUs
        if hasattr(self, 'Sections'):
            for obj in self.Sections:
                sub_case_folder = os.path.join(new_folder, str(obj.section_id))
                obj.set_case_folder(sub_case_folder)                        
        if hasattr(self, 'Summary'):
            self.Summary.set_model_attr(case_folder=self._case_folder)

    def set_runtime(self, runtime=None):
        """set runtime of the model
        runtime: a list of four values representing start, end, output interval
        and backup interval respectively
        """
        if runtime is None:
            runtime = [0, 3600, 3600, 3600]
        runtime = np.array(runtime)
        self.times = runtime
        runtime_str = ('{0}-start, {1}-end, {2}-output interval, '
                       '{3}-backup interval')
        runtime_str = runtime_str.format(*runtime)
        if hasattr(self, 'Summary'):
#            self.Summary.add_items('Runtime(s)', runtime_str)
            self.Summary.set_model_attr(run_time=self.times)
        return runtime_str

    def set_device_no(self, device_no=None):
        """set device no of the model
        device_no: int or a list of int corresponding to the number of sections 
        """
        if device_no is None:
            device_no = np.arange(self.num_of_sections)
        device_no = np.array(device_no)
        self.device_no = device_no
    
    def set_landcover(self, landcover_data):
        """ Set Landcover object with a Raster object or file
        """
        self.Landcover = Landcover(landcover_data, self.DEM)

    def add_user_defined_parameter(self, param_name, param_value):
        """ Add a grid-based user-defined parameter to the model
        param_name: (str) name the parameter and the input file name as well
        param_value: (scalar) or (numpy arary) with the same size of DEM array
        """
        if param_name not in InputHipims.__grid_files:
            InputHipims.__grid_files.append(param_name)
        self.attributes[param_name] = param_value
        print(param_name+ 'is added to the InputHipims object')
    
    def set_num_of_sections(self, num_of_sections):
        """ set the number of divided sections to run a case
        can transfer single-gpu to multi-gpu and the opposite way
        """
        self.num_of_sections = num_of_sections
        self.set_device_no()
        if num_of_sections==1: # to single GPU
            if hasattr(self, 'Sections'):
                del self.Sections
        else: # to multiple GPU
            self.__divide_grid()
            outline_boundary = self.Boundary.outline_boundary
            self.set_boundary_condition(outline_boundary=outline_boundary)
            self.set_gauges_position()
            self.Boundary._divide_domain(self)
        self.set_case_folder(self._case_folder)
        self.birthday = datetime.now()
        self.Summary.set_model_attr(num_GPU=self.num_of_sections)
#        time_str = self.birthday.strftime('%Y-%m-%d %H:%M:%S')

    def decomposite_domain(self, num_of_sections):
        """ divide a single-gpu case into a multi-gpu case
        """
        obj_mg = copy.deepcopy(self)
        obj_mg.num_of_sections = num_of_sections
        obj_mg.set_device_no()
        obj_mg.__divide_grid()
        obj_mg.set_case_folder() # set data_folders
        outline_boundary = obj_mg.Boundary.outline_boundary
        obj_mg.set_boundary_condition(outline_boundary=outline_boundary)
        obj_mg.set_gauges_position()
        obj_mg.Boundary._divide_domain(obj_mg)
        obj_mg.birthday = datetime.now()
        return obj_mg

#%%****************************************************************************
#************************Write input files*************************************
    def write_input_files(self, file_tag=None):
        """ Write input files
        To classify the input files and call functions needed to write each
            input files
        file_tag: str or list of str including
        """
        self._make_data_dirs()
        grid_files = InputHipims.__grid_files
        if file_tag is None or file_tag == 'all': # write all files
            file_tag_list = ['z', 'h', 'hU', 'precipitation',
                             'manning', 'sewer_sink',
                             'cumulative_depth', 'hydraulic_conductivity',
                             'capillary_head', 'water_content_diff',
                             'precipitation_mask', 'precipitation_source',
                             'boundary_condition', 'gauges_pos']
            if self.num_of_sections > 1:
                self.write_halo_file()
            self.write_mesh_file()
            self.write_runtime_file()
            self.write_device_file()
        elif type(file_tag) is str:
            file_tag_list = [file_tag]
        elif type(file_tag) is list:
            file_tag_list = file_tag
        else:
            print(file_tag_list)
            raise ValueError(('file_tag should be a string or a list of string'
                             'from the above list'))
        for one_file in file_tag_list:
            if one_file in grid_files: # grid-based files
                self.write_grid_files(one_file)
            elif one_file == 'boundary_condition':
                self.write_boundary_conditions()
            elif one_file == 'precipitation_source':
                self.write_rainfall_source()
            elif one_file == 'gauges_pos':
                self.write_gauges_position()
            else:
                raise ValueError(one_file+' is not recognized')

    def write_grid_files(self, file_tag, is_single_gpu=False):
        """Write grid-based files
        Public version for both single and multiple GPUs
        file_tag: the pure name of a grid-based file
        """
        self._make_data_dirs()
        grid_files = InputHipims.__grid_files
        if file_tag not in grid_files:
            raise ValueError(file_tag+' is not a grid-based file')
        if is_single_gpu or self.num_of_sections == 1:
            # write as single GPU even the num of sections is more than one
            self._write_grid_files(file_tag, is_multi_gpu=False)
        else:
            self._write_grid_files(file_tag, is_multi_gpu=True)
        readme_filename = os.path.join(self._case_folder,'readme.txt')
        self.Summary.to_json(readme_filename)
        print(file_tag+' created')

    def write_boundary_conditions(self):
        """ Write boundary condtion files
        if there are multiple domains, write in the first folder
            and copy to others
        """
        self._make_data_dirs()
        if self.num_of_sections > 1:  # multiple-GPU
            field_dir = self.Sections[0]._data_folders['field']
            file_names_list = self.__write_boundary_conditions(field_dir)
            self.__copy_to_all_sections(file_names_list)
        else:  # single-GPU
            field_dir = self._data_folders['field']
            self.__write_boundary_conditions(field_dir)
        readme_filename = os.path.join(self._case_folder,'readme.txt')
        self.Summary.to_json(readme_filename)
        print('boundary condition files created')

    def write_rainfall_source(self):
        """Write rainfall source data
        rainfall mask can be written by function write_grid_files
        """
        self._make_data_dirs()
        if hasattr(self, 'Rainfall'):
            rain_source = self.Rainfall.get_source_array()
        else:
            rain_source = self.attributes['precipitation_source']
        case_folder = self._case_folder
        num_of_sections = self.num_of_sections
        indep_f.write_rain_source(rain_source, case_folder, num_of_sections)
        readme_filename = os.path.join(self._case_folder,'readme.txt')
        self.Summary.to_json(readme_filename)

    def write_gauges_position(self, gauges_pos=None):
        """ Write the gauges position file
        Public version for both single and multiple GPUs
        """
        self._make_data_dirs()
        if gauges_pos is not None:
            self.set_gauges_position(np.array(gauges_pos))
        if self.num_of_sections > 1:  # multiple-GPU
            for obj_section in self.Sections:
                field_dir = obj_section._data_folders['field']
                obj_section.__write_gauge_ind(field_dir)
                obj_section.__write_gauge_pos(field_dir)
        else:  # single-GPU
            field_dir = self._data_folders['field']
            self.__write_gauge_pos(field_dir)
        readme_filename = os.path.join(self._case_folder,'readme.txt')
        self.Summary.to_json(readme_filename)
        print('gauges_pos.dat created')

    def write_halo_file(self):
        """ Write overlayed cell IDs
        """
        num_section = self.num_of_sections
        case_folder = self._case_folder
        file_name = os.path.join(case_folder, 'halo.dat')
        with open(file_name, 'w') as file2write:
            file2write.write("No. of Domains\n")
            file2write.write("%d\n" % num_section)
            for obj_section in self.Sections:
                file2write.write("#%d\n" % obj_section.section_id)
                overlayed_id = obj_section.overlayed_id
                for key in ['bottom_low', 'bottom_high',
                            'top_high', 'top_low']:
                    if key in overlayed_id.keys():
                        line_ids = overlayed_id[key]
                        line_ids = np.reshape(line_ids, (1, line_ids.size))
                        np.savetxt(file2write,
                                   line_ids, fmt='%d', delimiter=' ')
                    else:
                        file2write.write(' \n')
        print('halo.dat created')

    def write_mesh_file(self, is_single_gpu=False):
        """ Write mesh file DEM.txt, compatoble for both single and multiple
        GPU model
        """
        self._make_data_dirs()
        if is_single_gpu is True or self.num_of_sections == 1:
            file_name = os.path.join(self._data_folders['mesh'],
                                     'DEM.txt')
            self.DEM.write_asc(file_name)
        else:
            for obj_section in self.Sections:
                file_name = os.path.join(obj_section._data_folders['mesh'],
                                         'DEM.txt')
                obj_section.DEM.write_asc(file_name)
        readme_filename = os.path.join(self._case_folder,'readme.txt')
        self.Summary.to_json(readme_filename)
    
    def write_runtime_file(self, time_values=None):
        """ write times_setup.dat file
        """
        if time_values is None:
            time_values = self.times
        indep_f.write_times_setup(self._case_folder, self.num_of_sections,
                                  time_values)
        self.Summary.to_json(self._case_folder+'/readme.txt')
    
    def write_device_file(self, device_no=None):
        """Create device_setup.dat for choosing GPU number to run the model
        """
        if device_no is None:
            device_no = self.device_no
        indep_f.write_device_setup(self._case_folder, self.num_of_sections,
                                   device_no)

    def save_object(self, file_name):
        """ Save object as a pickle file
        """
        indep_f.save_as_dict(self, file_name)

#%%****************************************************************************
#******************************* Visualization ********************************
    def domain_show(self, figname=None, dpi=None, title='Domain Map',
                    **kwargs):
        """Show domain map of the object
        """
        obj_dem = copy.copy(self.DEM)
        if hasattr(self, 'Sections'):
            for obj_sub in self.Sections:
                overlayed_subs = obj_sub.overlayed_cell_subs_global
                obj_dem.array[overlayed_subs] = np.nan            
        fig, ax = obj_dem.mapshow(title=title, cax_str='DEM(m)', **kwargs)
        cell_subs = self.Boundary.cell_subs
        legends = []
        num = 0
        for cell_sub in cell_subs:
            rows = cell_sub[0]
            cols = cell_sub[1]
            X, Y = sub2map(rows, cols, self.DEM.header)
            ax.plot(X, Y, '.')
            legends.append('Boundary '+str(num))
            num = num+1
        legends[0] = 'Outline boundary'
        ax.legend(legends, edgecolor=None, facecolor=None, loc='best',
                  fontsize='x-small')
        if figname is not None:
            fig.savefig(figname, dpi=dpi)
        return fig, ax
    
    def plot_rainfall_map(self, figname=None, method='sum', **kw):
        """plot rainfall map within model domain
        """
        rain_source = self.attributes['precipitation_source']
        rain_mask = self.attributes['precipitation_mask']
        rain_mask = self.DEM.array*0+rain_mask
        rain_mask_obj = Raster(array=rain_mask, header=self.header)
        rain_map_obj = rp.get_spatial_map(rain_source, rain_mask_obj, figname,
                                          method, **kw)
        return rain_map_obj

    def plot_rainfall_curve(self, start_date=None, method='mean', **kw):
        """ Plot time series of average rainfall rate inside the model domain
        start_date: a datetime object to give the initial date and time of rain
        method: 'mean'|'max','min','mean'method to calculate gridded rainfall 
        over the model domain
        """
        rain_source = self.attributes['precipitation_source']
        rain_mask = self.attributes['precipitation_mask']
        rain_mask = np.array(rain_mask)
        rain_mask = self.DEM.array*0+rain_mask
        plot_data = rp.get_time_series(rain_source, rain_mask, start_date, 
                                    method=method)
        rp.plot_time_series(plot_data, method=method, **kw)
        return plot_data

#%%****************************************************************************
#*************************** Protected methods ********************************
    def _get_cell_subs(self, dem_array=None):
        """ To get valid_cell_subs and outline_cell_subs for the object
        To get the subscripts of each valid cell on grid
        Input arguments are for sub Hipims objects
        _valid_cell_subs
        _outline_cell_subs
        """
        if dem_array is None:
            dem_array = self.DEM.array
        valid_id, outline_id = indep_f._get_cell_id_array(dem_array)
        subs = np.where(~np.isnan(valid_id))
        id_vector = valid_id[subs]
        # sort the subscripts according to cell id values
        sorted_vectors = np.c_[id_vector, subs[0], subs[1]]
        sorted_vectors = sorted_vectors[sorted_vectors[:, 0].argsort()]
        self._valid_cell_subs = (sorted_vectors[:, 1].astype('int32'),
                                 sorted_vectors[:, 2].astype('int32'))
        subs = np.where(outline_id == 0) # outline boundary cell
        outline_id_vect = outline_id[subs]
        sorted_array = np.c_[outline_id_vect, subs[0], subs[1]]
        self._outline_cell_subs = (sorted_array[:, 1].astype('int32'),
                                   sorted_array[:, 2].astype('int32'))
    def __divide_grid(self):
        """
        Divide DEM grid to sub grids
        Create objects based on sub-class InputHipimsSub
        """
        if isinstance(self, InputHipimsSub):
            return 0  # do not divide InputHipimsSub objects, return a number
        else:
            if self.num_of_sections == 1:
                return 1 # do not divide if num_of_sections is 1
        num_of_sections = self.num_of_sections
        dem_header = self.header
        # subscripts of the split row [0, 1,...] from bottom to top
        split_rows = indep_f._get_split_rows(self.DEM.array, num_of_sections)
        array_local, header_local = \
            indep_f._split_array_by_rows(self.DEM.array, dem_header,
                                         split_rows)
        # to receive InputHipimsSub objects for sections
        Sections = []
        section_sequence = np.arange(num_of_sections)
        header_global = dem_header
        for i in section_sequence:  # from bottom to top
            case_folder = os.path.join(self._case_folder, str(i))
            # create a sub object of InputHipims
            sub_hipims = InputHipimsSub(array_local[i], header_local[i],
                                        case_folder, num_of_sections)
            # get valid_cell_subs on the global grid
            valid_cell_subs = sub_hipims._valid_cell_subs
            valid_subs_global = \
                 indep_f._cell_subs_convertor(valid_cell_subs, header_global,
                                      header_local[i], to_global=True)
            sub_hipims.valid_subs_global = valid_subs_global
            # record section sequence number
#            sub_hipims.section_id = i
            #get overlayed_id (top two rows and bottom two rows)
            top_h = np.where(valid_cell_subs[0] == 0)
            top_l = np.where(valid_cell_subs[0] == 1)
            bottom_h = np.where(
                valid_cell_subs[0] == valid_cell_subs[0].max()-1)
            bottom_l = np.where(valid_cell_subs[0] == valid_cell_subs[0].max())
            if i == 0: # the bottom section
                overlayed_id = {'top_high':top_h[0], 'top_low':top_l[0]}
            elif i == self.num_of_sections-1: # the top section
                overlayed_id = {'bottom_low':bottom_l[0],
                                'bottom_high':bottom_h[0]}
            else:
                overlayed_id = {'top_high':top_h[0], 'top_low':top_l[0],
                                'bottom_high':bottom_h[0],
                                'bottom_low':bottom_l[0]}
            sub_hipims.overlayed_id = overlayed_id
            all_ids = list(overlayed_id.values())
            all_ids = np.concatenate(all_ids).ravel()
            all_ids.sort()
            overlayed_cell_subs_global = (valid_subs_global[0][all_ids],
                                          valid_subs_global[1][all_ids])
            sub_hipims.overlayed_cell_subs_global = overlayed_cell_subs_global
            Sections.append(sub_hipims)
        # reset global var section_id of InputHipimsSub
        InputHipimsSub.section_id = 0
        self.Sections = Sections
        self._initialize_summary_obj()# get a Model Summary object

    def _get_vector_value(self, attribute_name, is_multi_gpu=True,
                          add_initial_water=True):
        """ Generate a single vector for values in each grid cell sorted based
        on cell IDs
        attribute_name: attribute names based on a grid
        Return:
            output_vector: a vector of values in global valid grid cells
                            or a list of vectors for each sub domain
        """
        # get grid value
        dem_shape = self.shape
        grid_values = np.zeros(dem_shape)
        if add_initial_water:
            add_value = 0.0001
        else:
            add_value = 0

        def add_water_on_io_cells(bound_obj, grid_values, source_key,
                                  add_value):
            """ add a small water depth/velocity to IO boundary cells
            """
            for ind_num in np.arange(bound_obj.num_of_bound):
                bound_source = bound_obj.data_table[source_key][ind_num]
                if bound_source is not None:
                    source_value = np.unique(bound_source[:, 1:])
                    # zero boundary conditions
                    if not (source_value.size == 1 and source_value[0] == 0):
                        cell_subs = bound_obj.cell_subs[ind_num]
                        grid_values[cell_subs] = add_value
            return grid_values
        # set grid value for the entire domain
        if attribute_name == 'z':
            grid_values = self.DEM.array
        elif attribute_name == 'h':
            grid_values = grid_values+self.attributes['h0']
            # traversal each boundary to add initial water
            grid_values = add_water_on_io_cells(self.Boundary, grid_values,
                                                'hSources', add_value)
        elif attribute_name == 'hU':
            grid_values0 = grid_values+self.attributes['hU0x']
            grid_values1 = grid_values+self.attributes['hU0y']
            grid_values1 = add_water_on_io_cells(self.Boundary, grid_values1,
                                                 'hUSources', add_value)
            grid_values = [grid_values0, grid_values1]
        elif attribute_name == 'precipitation_mask':
            if hasattr(self, 'Rainfall'):
                grid_values = self.Rainfall.get_mask_array()
            else:
                grid_values = grid_values+self.attributes[attribute_name]
        else:
            if hasattr(self, 'param_per_landcover'):
                arg_dicts = self.param_per_landcover
                if attribute_name in arg_dicts.keys():
                    arg_dict = arg_dicts[attribute_name]
                    grid_values = self.Landcover.to_grid_parameter(**arg_dict)
                else:
                    grid_values = grid_values+self.attributes[attribute_name]
            else:
                grid_values = grid_values+self.attributes[attribute_name]

        def grid_to_vect(grid_values, cell_subs):
            """ Convert grid values to 1 or 2 col vector values
            """
            if type(grid_values) is list:
                vector_value0 = grid_values[0][cell_subs]
                vector_value1 = grid_values[1][cell_subs]
                vector_value = np.c_[vector_value0, vector_value1]
            else:
                vector_value = grid_values[cell_subs]
            return vector_value
        #
        if is_multi_gpu: # generate vector value for multiple GPU
            output_vector = []
            for obj_section in self.Sections:
                cell_subs = obj_section.valid_subs_global
                vector_value = grid_to_vect(grid_values, cell_subs)
                output_vector.append(vector_value)
        else:
            output_vector = grid_to_vect(grid_values, self._valid_cell_subs)
        return output_vector

    def _get_boundary_id_code_array(self, file_tag='z'):
        """
        To generate a 4-col array of boundary cell id (0) and code (1~3)
        """
        bound_obj = self.Boundary
        output_array_list = []
        for ind_num in np.arange(bound_obj.num_of_bound):
            if file_tag == 'h':
                bound_code = bound_obj.data_table.h_code[ind_num]
            elif file_tag == 'hU':
                bound_code = bound_obj.data_table.hU_code[ind_num]
            else:
                bound_code = np.array([[2, 0, 0]]) # shape (1, 3)
            if bound_code.ndim < 2:
                bound_code = np.reshape(bound_code, (1, bound_code.size))
            cell_id = bound_obj.cell_id[ind_num]
            if cell_id.size > 0:
                bound_code_array = np.repeat(bound_code, cell_id.size, axis=0)
                id_code_array = np.c_[cell_id, bound_code_array]
                output_array_list.append(id_code_array)
        # add overlayed cells with [4, 0, 0]
        # if it is a sub section object, there should be attributes:
        # overlayed_id, and section_id
        if hasattr(self, 'overlayed_id'):
            cell_id = list(self.overlayed_id.values())
            cell_id = np.concatenate(cell_id, axis=0)
            bound_code = np.array([[4, 0, 0]]) # shape (1, 3)
            bound_code_array = np.repeat(bound_code, cell_id.size, axis=0)
            id_code_array = np.c_[cell_id, bound_code_array]
            output_array_list.append(id_code_array)
        output_array = np.concatenate(output_array_list, axis=0)
        # when unique the output array according to cell id
        # keep the last occurrence rather than the default first occurrence
        output_array = np.flipud(output_array) # make the IO boundaries first
        _, ind = np.unique(output_array[:, 0], return_index=True)
        output_array = output_array[ind]
        return output_array

    def _initialize_summary_obj(self):
        """ Initialize the model summary object
        """
        summary_obj = Summary(self)
        self.Summary = summary_obj

    def _write_grid_files(self, file_tag, is_multi_gpu=True):
        """ Write input files consistent with the DEM grid
        Private function called by public function write_grid_files
        file_name: includes ['h','hU','precipitation_mask',
                             'manning','sewer_sink',
                             'cumulative_depth', 'hydraulic_conductivity',
                             'capillary_head', 'water_content_diff']
        """
        if is_multi_gpu is True:  # write for multi-GPU, use child object
            vector_value_list = self._get_vector_value(file_tag, is_multi_gpu)
            for obj_section in self.Sections:
                vector_value = vector_value_list[obj_section.section_id]
                cell_id = np.arange(vector_value.shape[0])
                cells_vect = np.c_[cell_id, vector_value]
                file_name = os.path.join(obj_section._data_folders['field'],
                                         file_tag+'.dat')
                if file_tag == 'precipitation_mask':
                    bounds_vect = None
                else:
                    bounds_vect = \
                        obj_section._get_boundary_id_code_array(file_tag)
                indep_f._write_two_arrays(file_name, cells_vect, bounds_vect)
        else:  # single GPU, use global object
            file_name = os.path.join(self._data_folders['field'],
                                     file_tag+'.dat')
            vector_value = self._get_vector_value(file_tag, is_multi_gpu=False)
            cell_id = np.arange(vector_value.shape[0])
            cells_vect = np.c_[cell_id, vector_value]
            if file_tag == 'precipitation_mask':
                bounds_vect = None
            else:
                bounds_vect = self._get_boundary_id_code_array(file_tag)
            indep_f._write_two_arrays(file_name, cells_vect, bounds_vect)
        return None

    def _make_data_dirs(self):
        """ Create folders in current device
        """
        if hasattr(self, 'Sections'):
            for obj_section in self.Sections:
                indep_f._create_io_folders(obj_section.get_case_folder(),
                                           make_dir=True)
        else:
            indep_f._create_io_folders(self._case_folder, make_dir=True)
    
    def _dict2grid(self, mask_dict):
        """Convert mask_dict to a grid array with the same shape of DEM
        """
        num_values = mask_dict['value'].size
        grid_array = np.zeros(self.DEM.shape).astype(mask_dict['value'].dtype)
        for i in np.arange(num_values):
            grid_array[mask_dict['index'][i]] = mask_dict['value'][i]
        return grid_array
        
#------------------------------------------------------------------------------
#*************** Private methods only for the parent class ********************
#------------------------------------------------------------------------------
    def __write_boundary_conditions(self, field_dir, file_tag='both'):
        """ Write boundary condition source files,if hU is given as flow
        timeseries, convert flow to hUx and hUy.
        Private function to call by public function write_boundary_conditions
        file_tag: 'h', 'hU', 'both'
        h_BC_[N].dat, hU_BC_[N].dat
        if hU is given as flow timeseries, convert flow to hUx and hUy
        """
        obj_boundary = self.Boundary
        file_names_list = []
        fmt_h = ['%g', '%g']
        fmt_hu = ['%g', '%g', '%g']
        # write h_BC_[N].dat
        if file_tag in ['both', 'h']:
            h_sources = obj_boundary.data_table['hSources']
            ind_num = 0
            for i in np.arange(obj_boundary.num_of_bound):
                h_source = h_sources[i]
                if h_source is not None:
                    file_name = os.path.join(field_dir, 'h_BC_'+str(ind_num)+'.dat')
                    np.savetxt(file_name, h_source, fmt=fmt_h, delimiter=' ')
                    ind_num = ind_num+1
                    file_names_list.append(file_name)
        # write hU_BC_[N].dat
        if file_tag in ['both', 'hU']:
            hU_sources = obj_boundary.data_table['hUSources']
            ind_num = 0
            for i in np.arange(obj_boundary.num_of_bound):
                hU_source = hU_sources[i]
                cell_subs = obj_boundary.cell_subs[i]
                if hU_source is not None:
                    file_name = os.path.join(field_dir, 'hU_BC_'+str(ind_num)+'.dat')
                    if hU_source.shape[1] == 2:
                        # flow is given rather than speed
                        boundary_slope = np.polyfit(cell_subs[0],
                                                    cell_subs[1], 1)
                        theta = np.arctan(boundary_slope[0])
                        boundary_length = cell_subs[0].size* \
                                          self.DEM.header['cellsize']
                        hUx = hU_source[:, 1]*np.cos(theta)/boundary_length
                        hUy = hU_source[:, 1]*np.sin(theta)/boundary_length
                        hU_source = np.c_[hU_source[:, 0], hUx, hUy]
                        print('Flow series on boundary '+str(i)+
                              ' is converted to velocities')
                        print('Theta = '+'{:.3f}'.format(theta/np.pi)+'*pi')
                    np.savetxt(file_name, hU_source, fmt=fmt_hu, delimiter=' ')
                    ind_num = ind_num+1
                    file_names_list.append(file_name)
        return file_names_list

    def __write_gauge_pos(self, file_folder):
        """write monitoring gauges
        gauges_pos.dat
        file_folder: folder to write file
        gauges_pos: 2-col numpy array of X and Y coordinates
        """
        gauges_pos = self.attributes['gauges_pos']
        file_name = os.path.join(file_folder, 'gauges_pos.dat')
        fmt = ['%g %g']
        fmt = '\n'.join(fmt*gauges_pos.shape[0])
        gauges_pos_str = fmt % tuple(gauges_pos.ravel())
        with open(file_name, 'w') as file2write:
            file2write.write(gauges_pos_str)
        return file_name

    def __write_gauge_ind(self, file_folder):
        """write monitoring gauges index for mult-GPU sections
        gauges_ind.dat
        file_folder: folder to write file
        gauges_ind: 1-col numpy array of index values
        """
        gauges_ind = self.attributes['gauges_ind']
        file_name = os.path.join(file_folder, 'gauges_ind.dat')
        fmt = ['%g']
        fmt = '\n'.join(fmt*gauges_ind.shape[0])
        gauges_ind_str = fmt % tuple(gauges_ind.ravel())
        with open(file_name, 'w') as file2write:
            file2write.write(gauges_ind_str)
        return file_name

    def __copy_to_all_sections(self, file_names):
        """ Copy files that are the same in each sections
        file_names: (str) files written in the first seciton [0]
        boundary source files: h_BC_[N].dat, hU_BC_[N].dat
        rainfall source files: precipitation_source_all.dat
        gauges position file: gauges_pos.dat
        """
        if type(file_names) is not list:
            file_names = [file_names]
        for i in np.arange(1, self.num_of_sections):
            field_dir = self.Sections[i]._data_folders['field']
            for file in file_names:
                shutil.copy2(file, field_dir)
    
    def _setup_by_files(self, data_path):
        """Read files to setup hipims input object
        DEM, landcover, rain_mask, endswith '.gz', '.asc', or '.tif'
        rain_source.csv
        """
        data_path = self.data_path
        ras_file = indep_f._check_raster_exist(os.path.join(data_path, 'DEM'))
        if ras_file is not None:
            self.DEM = Raster(ras_file)
            print(ras_file+ ' read')
        ras_file = indep_f._check_raster_exist(os.path.join(data_path,
                                                            'landcover'))        
        if ras_file is not None:
            self.Landcover = Landcover(ras_file, dem_ras=self.DEM)
            print(ras_file+ ' read')
        
        mask_file = indep_f._check_raster_exist(os.path.join(data_path,
                                                         'rain_mask'))
        if mask_file is None:
            mask_file = 0
        source_file = os.path.join(data_path, 'rain_source.csv')
        if not os.path.isfile(source_file):
            source_file = np.array([[0, 0], [1, 0]])
        self.Rainfall = Rainfall(mask_file, source_file, dem_ras=self.DEM)

#%%****************************************************************************
#************************sub-class definition**********************************
class InputHipimsSub(InputHipims):
    """object for each section, child class of InputHipims
    Attributes:
        sectionNO: the serial number of each section
        _valid_cell_subs: (tuple, int) two numpy array indicating rows and cols
        of valid cells on the local grid
        valid_cell_subsOnGlobal: (tuple, int) two numpy array indicating rows
        and cols of valid cells on the global grid
        shared_cells_id: 2-row shared Cells id on a local grid
        case_folder: input folder of each section
        _outline_cell_subs: (tuple, int) two numpy array indicating rows and 
        cols of valid cells on a local grid
    """
    section_id = 0
    def __init__(self, dem_array, header, case_folder, num_of_sections):
        self.section_id = InputHipimsSub.section_id
        InputHipimsSub.section_id = self.section_id+1
        dem_data = Raster(array=dem_array, header=header)
        super().__init__(dem_data, num_of_sections, case_folder)

#%%****************************************************************************
#********************************Static method*********************************

def main():
    print('Class to setup input data')

if __name__=='__main__':
    main()