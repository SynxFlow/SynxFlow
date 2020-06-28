#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
indep_functions
To do:
    non-object-based independent functions to support hipims IO
-----------------    
Created on Thu Apr 23 11:45:24 2020

@author: Xiaodong Ming
"""
import copy
import gzip
import pickle
import os
import shutil
import scipy.signal
import numpy as np
from .rainfall_processing import _check_rainfall_rate_values

def load_object(file_name):
    """ Read a pickle file as an InputHipims/OutputHipims object
    """
    #read an object file
    try:
        with gzip.open(file_name, 'rb') as input_file:
            obj = pickle.load(input_file)
    except:
        with open(file_name, 'rb') as input_file:
            obj = pickle.load(input_file)
    print(file_name+' loaded')
    return obj

def save_object(obj, file_name, compression=True):
    """ Save an InputHipims/OutputHipims object to a pickle file 
    """
    # Overwrites any existing file.
    if compression:
        with gzip.open(file_name, 'wb') as output_file:
            pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)
    else:
        with open(file_name, 'wb') as output_file:
            pickle.dump(obj, output_file, pickle.HIGHEST_PROTOCOL)
    print(file_name+' has been saved')

def save_as_dict(obj, file_name):
    """Save all attributes of an object to a pickle
    """
    obj_dict = copy.copy(obj.__dict__)
    
    obj_dict['DEM'] = obj.DEM.__dict__
    obj_dict['Summary'] = obj.Summary.to_dict()
    obj_dict['Boundary'] = obj.Boundary.__dict__
    if hasattr(obj, 'Sections'):
        obj_dict.pop('Sections')
    if hasattr(obj, 'Rainfall'):
        obj_dict['Rainfall'] = obj.Rainfall.__dict__
    if hasattr(obj, 'Landcover'):
        obj_dict['Landcover'] = obj.Landcover.__dict__
    save_object(obj_dict, file_name, compression=True)

def clean_output(case_folder, num_of_sections, file_tag='*'):
    """ delete contents in output folder(s)
    """
    if case_folder[-1]!='/':
            case_folder = case_folder+'/'
    if num_of_sections==1:
        files_to_remove = case_folder+'/output/'+file_tag
        os.system('rm '+files_to_remove)
    else:    
        for i in range(num_of_sections):
            files_to_remove = case_folder+str(i)+'/output/'+file_tag
            os.system('rm '+files_to_remove)

#%% ***************************************************************************
# *************************Public functions************************************
def write_times_setup(case_folder=None, num_of_sections=1, time_values=None):
    """
    Generate a times_setup.dat file. The file contains numbers representing
    the start time, end time, output interval, and backup interval in seconds
    time_values: array or list of int/float, representing time in seconds,
        default values are [0, 3600, 1800, 3600]
    """
    case_folder = _check_case_folder(case_folder)
    if time_values is None:
        time_values = np.array([0, 3600, 1800, 3600])
    time_values = np.array(time_values)
    time_values = time_values.reshape((1, time_values.size))
    if num_of_sections == 1:
        np.savetxt(case_folder+'/input/times_setup.dat', time_values, fmt='%g')
    else:
        np.savetxt(case_folder+'/times_setup.dat', time_values, fmt='%g')
    print('times_setup.dat created')

def write_device_setup(case_folder=None,
                       num_of_sections=1, device_values=None):
    """
    Generate a device_setup.dat file. The file contains numbers representing
    the GPU number for each section
    case_folder: string, the path of model
    num_of_sections: int, the number of GPUs to use
    device_values: array or list of int, representing the GPU number
    """
    case_folder = _check_case_folder(case_folder)
    if device_values is None:
        device_values = np.array(range(num_of_sections))
    device_values = np.array(device_values)
    device_values = device_values.reshape((1, device_values.size))
    if num_of_sections == 1:
        np.savetxt(case_folder+'/input/device_setup.dat',
                   device_values, fmt='%g')
    else:
        np.savetxt(case_folder+'/device_setup.dat', device_values, fmt='%g')
    print('device_setup.dat created')

def write_rain_source(rain_source, case_folder=None, num_of_sections=1):
    """ Write rainfall sources [Independent function from hipims class]
    rain_source: numpy array, The 1st column is time in seconds, the 2nd
        towards the end columns are rainfall rate in m/s for each source ID in
        rainfall mask array
    if for multiple GPU, then copy the rain source file to all domain folders
    case_folder: string, the path of model
    """
    rain_source = np.array(rain_source)
    # check rainfall source value to avoid very large raifall rates
    _ = _check_rainfall_rate_values(rain_source)
    case_folder = _check_case_folder(case_folder)
    fmt1 = '%g'  # for the first col: times in seconds
    fmt2 = '%.8e'  # for the rest array for rainfall rate m/s
    num_mask_cells = rain_source.shape[1]-1
    format_spec = [fmt2]*num_mask_cells
    format_spec.insert(0, fmt1)
    if num_of_sections == 1: # single GPU
        file_name = case_folder+'input/field/precipitation_source_all.dat'
    else: # multi-GPU
        file_name = case_folder+'0/input/field/precipitation_source_all.dat'
    with open(file_name, 'w') as file2write:
        file2write.write("%d\n" % num_mask_cells)
        np.savetxt(file2write, rain_source, fmt=format_spec, delimiter=' ')
    if num_of_sections > 1:
        for i in np.arange(1, num_of_sections):
            field_dir = case_folder+str(i)+'/input/field/'
            shutil.copy2(file_name, field_dir)
    print('precipitation_source_all.dat created')

def _write_two_arrays(file_name, id_values, bound_id_code=None):
    """Write two arrays: cell_id-value pairs and bound_id-bound_code pairs
    Inputs:
        file_name :  the full file name including path
        id_values: valid cell ID - value pair
        bound_id_code: boundary cell ID - codes pair. If bound_id_code is not
            given, then the second part of the file won't be written (only
            the case for precipitatin_mask.dat)
    """
    if not file_name.endswith('.dat'):
        file_name = file_name+'.dat'
    if id_values.shape[1] == 3:
        fmt = ['%d %g %g']
    elif id_values.shape[1] == 2:
        fmt = ['%d %g']
    else:
        raise ValueError('Please check the shape of the 1st array: id_values')
    fmt = '\n'.join(fmt*id_values.shape[0])
    id_values_str = fmt % tuple(id_values.ravel())
    if bound_id_code is not None:
        fmt = ['%-12d %2d %2d %2d']
        fmt = '\n'.join(fmt*bound_id_code.shape[0])
        bound_id_code_str = fmt % tuple(bound_id_code.ravel())
    with open(file_name, 'w') as file2write:
        file2write.write("$Element Number\n")
        file2write.write("%d\n" % id_values.shape[0])
        file2write.write("$Element_id  Value\n")
        file2write.write(id_values_str)
        if bound_id_code is not None:
            file2write.write("\n$Boundary Numbers\n")
            file2write.write("%d\n" % bound_id_code.shape[0])
            file2write.write("$Element_id  Value\n")
            file2write.write(bound_id_code_str)

#%% static method for InputHipims
def _create_io_folders(case_folder, make_dir=False):
    """ create Input-Output path for a Hipims case
        (compatible for single/multi-GPU)
    Return:
      dir_input, dir_output, dir_mesh, dir_field
    """
    folder_name = case_folder
    dir_input = os.path.join(folder_name, 'input')
    dir_output = os.path.join(folder_name, 'output')
    if not os.path.exists(dir_output) and make_dir:
        os.makedirs(dir_output)
    if not os.path.exists(dir_input) and make_dir:
        os.makedirs(dir_input)
    dir_mesh = os.path.join(dir_input, 'mesh')
    if not os.path.exists(dir_mesh) and make_dir:
        os.makedirs(dir_mesh)
    dir_field = os.path.join(dir_input, 'field')
    if not os.path.exists(dir_field) and make_dir:
        os.makedirs(dir_field)
    data_folders = {'input':dir_input, 'output':dir_output,
                    'mesh':dir_mesh, 'field':dir_field}
    return data_folders

def _split_array_by_rows(input_array, header, split_rows, overlayed_rows=2):
    """ Clip an array into small ones according to the seperating rows
    input_array : the DEM array
    header : the DEM header
    split_rows : a list of row subscripts to split the array
    Split from bottom to top
    Return array_local, header_local: lists to store local DEM array and header
    """
    header_global = header
    end_row = header_global['nrows']-1
    overlayed_rows = 1
    array_local = []
    header_local = []
    section_sequence = np.arange(len(split_rows)+1)
    for i in section_sequence:  # from bottom to top
        section_id = i
        if section_id == section_sequence.max(): # the top section
            start_row = 0
        else:
            start_row = split_rows[i]-overlayed_rows
        if section_id == 0: # the bottom section
            end_row = header_global['nrows']-1
        else:
            end_row = split_rows[i-1]+overlayed_rows-1
        sub_array = input_array[start_row:end_row+1, :]
        array_local.append(sub_array)
        sub_header = header_global.copy()
        sub_header['nrows'] = sub_array.shape[0]
        sub_yllcorner = (header_global['yllcorner']+
                         (header_global['nrows']-1-end_row)*
                         header_global['cellsize'])
        sub_header['yllcorner'] = sub_yllcorner
        header_local.append(sub_header)
    return array_local, header_local

def _get_split_rows(input_array, num_of_sections):
    """ Split array by the number of valid cells (not NaNs) on each rows
    input_array : an array with some NaNs
    num_of_sections : (int) number of sections that the array to be splited
    return split_rows : a list of row subscripts to split the array
    Split from bottom to top
    """
    valid_cells = ~np.isnan(input_array)
    # valid_cells_count by rows
    valid_cells_count = np.sum(valid_cells, axis=1)
    valid_cells_count = np.cumsum(valid_cells_count)
    split_rows = []  # subscripts of the split row [0, 1,...]
    total_valid_cells = valid_cells_count[-1]
    for i in np.arange(num_of_sections-1): 
        num_section_cells = total_valid_cells*(i+1)/num_of_sections
        split_row = np.sum(valid_cells_count<=num_section_cells)-1
        split_rows.append(split_row)
    # sort from bottom to top
    split_rows.sort(reverse=True)
    return split_rows

def _get_cell_id_array(dem_array):
    """ to generate two arrays with the same size of dem_array:
    1. valid_id: to store valid cell id values (sequence number )
        starting from 0, from bottom, left to right, top
    2. outline_id: to store valid cell id on the boundary cells
    valid_id, outline_id = __get_cell_id_array(dem_array)
    """
    # convert DEM to a two-value array: NaNs and Ones
    # and flip up and down
    dem_array_flip = np.flipud(dem_array*0+1)
    # Return the cumulative sum of array elements over a given axis
    # treating NaNs) as zero.
    nancumsum_vector = np.nancumsum(dem_array_flip)
    # sequence number of valid cells: 0 to number of cells-1
    valid_id = nancumsum_vector-1
    # reshape as an array with the same size of DEM
    valid_id = np.reshape(valid_id, np.shape(dem_array_flip))
    # set NaN cells as NaNs
    valid_id[np.isnan(dem_array_flip)] = np.nan
    valid_id = np.flipud(valid_id)
    # find the outline boundary cells
    array_for_outline = dem_array*0
    array_for_outline[np.isnan(dem_array)] = -1
    h_hv = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    # Convolve two array_for_outline arrays
    ind_array = scipy.signal.convolve2d(array_for_outline, h_hv, mode='same')
    ind_array[ind_array < 0] = np.nan
    ind_array[0, :] = np.nan
    ind_array[-1, :] = np.nan
    ind_array[:, 0] = np.nan
    ind_array[:, -1] = np.nan
    # extract the outline cells by a combination
    ind_array = np.isnan(ind_array) & ~np.isnan(dem_array)
    # boundary cells with valid cell id are extracted
    outline_id = dem_array*0-2 # default inner cells value:-2
    outline_id[ind_array] = 0 # outline cells:0
    return valid_id, outline_id

def _cell_subs_convertor(input_cell_subs, header_global,
                         header_local, to_global=True):
    """
    Convert global cell subs to divided local cell subs or the otherwise
    and return output_cell_subs, only rows need to be changed
    input_cell_subs : (tuple) input rows and cols of a grid
    header_global : head information of the global grid
    header_local : head information of the local grid
    to_global : logical values, True (local to global) or
                                False(global to local)
    Return:
        output_cell_subs: (tuple) output rows and cols of a grid
    """
    # X and Y coordinates of the centre of the first cell
    y00_centre_global = header_global['yllcorner']+\
                         (header_global['nrows']+0.5)*header_global['cellsize']
    y00_centre_local = header_local['yllcorner']+\
                        (header_local['nrows']+0.5)*header_local['cellsize']
    row_gap = (y00_centre_global-y00_centre_local)/header_local['cellsize']
    row_gap = round(row_gap)
    rows = input_cell_subs[0]
    cols = input_cell_subs[1]
    if to_global:
        rows = rows+row_gap
        # remove subs out of range of the global DEM
        ind = np.logical_and(rows >= 0, rows < header_global['nrows'])
    else:
        rows = rows-row_gap
        # remove subs out of range of the global DEM
        ind = np.logical_and(rows >= 0, rows < header_local['nrows'])
    rows = rows.astype(cols.dtype)
    rows = rows[ind]
    cols = cols[ind]
    output_cell_subs = (rows, cols)
    return output_cell_subs

#%% =======================Value check functions===============================
def _check_case_folder(case_folder):
    """ check the format of case folder
    """
    if case_folder is None:
        case_folder = os.getcwd()
    if not case_folder.endswith('/'):
        case_folder = case_folder+'/'
    return case_folder

def _check_raster_exist(file_tag):
    """check the existence of raster files .gz, .asc, .tif
    file_tag: 'DEM', 'rain_mask', 'landcover'
    """
    if os.path.isfile(file_tag+'.gz'):
        file_name = file_tag+'.gz'
    elif os.path.isfile(file_tag+'.asc'):
        file_name = file_tag+'.asc'
    elif os.path.isfile(file_tag+'.tif'):
        file_name = file_tag+'.tif'
    else:
        file_name = None
    return file_name

def _mask2dict(obj_mask, new_header=None):
    """Convert a mask Raster object to a dictionary
    """
    if new_header is not None:
        obj_mask = obj_mask.assign_to(new_header)
    array = obj_mask.array
    v_unique = np.unique(array[~np.isnan(array)])
    index = []
    for onevalue in v_unique:
        ind = np.where(array==onevalue)
        index.append(ind)
    mask_dict = {'value':v_unique, 'index':index}
    return mask_dict

def _dict2grid(mask_dict, array_shape):
    """Convert mask_dict to a grid array with the shape given
    """
    num_values = mask_dict['value'].size
    grid_array = np.zeros(array_shape).astype(mask_dict['value'].dtype)
    for i in np.arange(num_values):
        grid_array[mask_dict['index'][i]] = mask_dict['value'][i]
    return grid_array