#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Xiaodong Ming

"""
Boundary
========

To do:
    * Define boundary conditions for hipims model

-----------

"""
# Created on Tue Mar 31 16:05:27 2020
import warnings
import numpy as np
import pandas as pd
import matplotlib.patches as mplP
from . import spatial_analysis as sp
#%% boundary class definition
class Boundary(object):
    """
    Class for boundary conditions

    Attributes:

        num_of_bound: number of boundaries

        type: a list of string 'open', 'rigid', 'fall'
                'open': timeseries of boundary depth/discharge/velocity can 
                    be given for this type. If no given timeseries data,
                    water will flow out flatly
                'rigid': no outlet
                'fall': water flow out like a fall, a fix zero water depth and
                    velocities will be given

        extent: (2-col numpy array) poly points to define the extent of a
                IO boundary. If extent is not given, then the boundary is
                the domain outline

        hSources: a two-col numpy array. The 1st col is time(s). The 2nd
                col is water depth(m)

        hUSources: a two-col numpy array. The 1st col is time(s). The 2nd
                col is discharge(m3/s) or a three-col numpy array, the 2nd
                col and the 3rd col are velocities(m/s) in x and y
                direction, respectively.

        h_code: 3-element int to define the type of depth boundary

        hU_code: 3-element int to define th type of velocity boundary

        description: (str) description of a boundary

    """
    def __init__(self, boundary_list=None, outline_boundary='fall'):
        """Initialise the object

        Args:
            boundary_list: (list of dicts), each dict contain keys (polyPoints,
                type, h, hU) to define a IO boundary's position, type, and
                Input-Output (IO) sources timeseries. Keys including:

                1.polyPoints is a numpy array giving X(1st col) and Y(2nd col)
                    coordinates of points to define the position of a boundary.
                    A bound without polyPoints is regarded as the outline_boundary.

                2.type: string, type of the boundary
                    'open': timeseries of boundary depth/discharge/velocity
                            can be given for this type. If no given timeseries
                            data, water will flow out flatly
                    'rigid': water cannot flow in or out
                    'fall': water flow out like a fall, a fix zero water depth
                            and velocities will be given

                3.h: a two-col numpy array. The 1st col is time(s). The 2nd col
                     is water depth(m)

                4.hU: a two-col numpy array. The 1st col is time(s). The 2nd
                    col is discharge(m3/s) or a three-col numpy array, the 2nd
                    col and the 3rd col are velocities(m/s) in x and y
                    direction, respectively.

            outline_boundary: (str) 'open'|'rigid', default outline boundary is
                open and both h and hU are set as zero
            if h or hU is given, then the boundary type is set as 'open' in 
                function _setup_boundary_data_table
        """
        data_table = _setup_boundary_data_table(boundary_list, outline_boundary)
        data_table = _get_boundary_code(data_table)
        num_of_bound = data_table.shape[0]
        self.data_table = data_table
        self.num_of_bound = num_of_bound
        self.h_sources = data_table['hSources']
        self.hU_sources = data_table['hUSources']
        self.boundary_list = boundary_list
        self.outline_boundary = outline_boundary
        self.cell_subs = None
        self.cell_id = None

    def print_summary(self):
        """Print the summary information
        
        """
        print('Number of boundaries: '+str(self.num_of_bound))
        for n in range(self.num_of_bound):
            if self.cell_subs is not None:
                num_cells = self.cell_subs[n][0].size
                description = self.data_table.description[n] \
                                 + ', number of cells: '+str(num_cells)
                print(str(n)+'. '+description)

    def get_summary(self):
        """ Get summary information strings
        
        """
        summary_dict = {}
        summary_dict['Number of boundaries'] = str(self.num_of_bound)
        summary_str = []
        for n in np.arange(self.num_of_bound):
            if self.cell_subs is not None:
                num_cells = self.cell_subs[n][0].size
                description = self.data_table.description[n] \
                                 + ', number of cells: '+str(num_cells)
                summary_str.append(str(n)+'. '+description)
        summary_dict['Boundary details'] = summary_str
        return summary_dict

    def _fetch_boundary_cells(self, valid_subs, outline_subs, dem_header):
        """ To get the subsripts and id of boundary cells on the domain grid
        
        valid_subs, outline_subs, dem_header are from hipims object
            _valid_cell_subs, _outline_cell_subs
        cell_subs: (tuple)subscripts of outline boundary cells
        cell_id: (numpy vector)valid id of outline boundary cells
        """
        # to get outline cell id based on _outline_cell_subs
        vector_id = np.arange(valid_subs[0].size)
        nrows = dem_header['nrows']
        ncols = dem_header['ncols']
        cellsize = dem_header['cellsize']
        xllcorner = dem_header['xllcorner']
        yllcorner = dem_header['yllcorner']
        grid_cell_id = np.zeros((nrows, ncols))
        grid_cell_id[valid_subs] = vector_id
        outline_id = grid_cell_id[outline_subs]
        outline_id = outline_id.astype('int64')
        # to get boundary cells based on the spatial extent of each boundary
        bound_cell_x = xllcorner+(outline_subs[1]+0.5)*cellsize
        bound_cell_y = yllcorner+(nrows-outline_subs[0]-0.5) *cellsize
        n = 1 # sequence number of boundaries
        data_table = self.data_table
        cell_subs = []
        cell_id = []
        for n in range(data_table.shape[0]):
            if data_table.extent[n] is None: #outline boundary
                dem_extent = sp.header2extent(dem_header)
                polyPoints = sp.extent2shape_points(dem_extent)
            elif len(data_table.extent[n]) == 2:
                xyv = data_table.extent[n]
                polyPoints = sp.extent2shape_points([np.min(xyv[:, 0]),
                                                     np.max(xyv[:, 0]),
                                                     np.min(xyv[:, 1]),
                                                     np.max(xyv[:, 1])])
            else:
                polyPoints = data_table.extent[n]
            poly = mplP.Polygon(polyPoints, closed=True)
            bound_cell_xy = np.array([bound_cell_x, bound_cell_y])
            bound_cell_xy = np.transpose(bound_cell_xy)
            ind1 = poly.contains_points(bound_cell_xy)
            row = outline_subs[0][ind1]
            col = outline_subs[1][ind1]
            cell_id.append(outline_id[ind1])
            cell_subs.append((row, col))
        self.cell_subs = cell_subs
        self.cell_id = cell_id
        self.cell_subs_wet_io = _find_wet_io_cells(self, source_key='hUSources')
    
    def _convert_flow2velocity(self, dem_obj):
        """ Convert 2-col flow timeseries to 3-col velocity timeseries
        in datatable
        """
        hU_sources = self.data_table['hUSources']
        for i in np.arange(self.num_of_bound):
            hU_source = hU_sources[i]
            cell_subs = self.cell_subs[i] # row and col
            if hU_source is not None:
                if hU_source.shape[1] == 2:
                    if cell_subs[0].size == 1:
                        warnings.warn('Only one cell on boundary '+str(i)+
                              ', you should better convert flow to velocities by yourself')
                # flow is given, no velocity
                    theta = _get_bound_normal(cell_subs, dem_obj)
                    boundary_length = cell_subs[0].size*dem_obj.header['cellsize']
                    hUx = hU_source[:, 1]*np.cos(theta)/boundary_length
                    hUy = hU_source[:, 1]*np.sin(theta)/boundary_length
                    hU_source = np.c_[hU_source[:, 0], hUx, hUy]
                    self.data_table['hUSources'][i] = hU_source
                    print('Flow series on boundary '+str(i)+
                              ' is converted to velocities')
                    print('Theta = '+'{:.2f}'.format(theta/np.pi*180)+'degree')
        self.hU_sources = self.data_table['hUSources']

    def _divide_domain(self, hipims_obj):
        """ Create Boundary objects for each sub-domain
        
        IF hipims_obj has sub sections
        """
        boundary_list = hipims_obj.Boundary.boundary_list
        outline_boundary = hipims_obj.Boundary.outline_boundary
        header_global = hipims_obj.header
        outline_subs = hipims_obj._outline_cell_subs
        for i in range(hipims_obj.num_of_sections):
            obj_section = hipims_obj.Sections[i]
            header_local = obj_section.header
            # convert global subscripts to local
            outline_subs_local = _cell_subs_convertor(
                outline_subs, header_global, header_local, to_global=False)
            valid_subs_local = obj_section._valid_cell_subs
            bound_obj = Boundary(boundary_list, outline_boundary)
            bound_obj._fetch_boundary_cells(
                valid_subs_local, outline_subs_local, header_local)
            obj_section.Boundary = bound_obj
#            summary_str = bound_obj.get_summary()
#            obj_section.Summary.add_items('Boundary conditions', summary_str)

#%%
# private function called by Class Boundary
def _setup_boundary_data_table(boundary_list, outline_boundary='open'):
    """ Initialize boundary data table based on boundary_list
    Add attributes type, extent, hSources, hUSources
    boundary_list: (list) of dict with keys polyPoints, h, hU
    outline_boundary: (str) 'open', 'rigid', 'fall'
    A bound without polyPoints is regarded as the outline_boundary
    If h or hU is given, then the boundary type is set as 'open', which means
     'h' and 'hU' have priority to 'type'
    """
    data_table = pd.DataFrame(columns=['type', 'extent',
                                       'hSources', 'hUSources',
                                       'h_code', 'hU_code', 'name'])
    # set default outline boundary [0]
    if outline_boundary == 'fall':
        hSources = np.array([[0, 0], [1, 0]])
        hUSources = np.array([[0, 0, 0], [1, 0, 0]])
        data_table = data_table.append({'type':'fall', 'extent':None,
                                        'hSources':hSources,
                                        'hUSources':hUSources},
                                       ignore_index=True)
    elif outline_boundary == 'rigid':
        data_table = data_table.append({'type':'rigid', 'extent':None,
                                        'hSources':None, 'hUSources':None},
                                       ignore_index=True)
    elif outline_boundary == 'open':
        data_table = data_table.append({'type':'open', 'extent':None,
                                        'hSources':None, 'hUSources':None},
                                       ignore_index=True)
    else:
        raise ValueError("outline_boundary must be fall, open or rigid!")
    data_table.name[0] = 'Outline boundary'
    # convert boundary_list to a dataframe
    if boundary_list is None:
        boundary_list = []
    for one_bound in boundary_list:
        # Only a bound with polyPoints can be regarded as a boundary
        # Otherwise it will be treated as the outline boundary
        bound_ind = data_table.shape[0]  # bound index
            # set extent
        if 'polyPoints' in one_bound.keys():
            bound_extent = np.array(one_bound['polyPoints'])
            data_table = data_table.append({'extent':bound_extent, }, 
                                           ignore_index=True)
        else:
            bound_ind = 0
            print('polyPoints is not given in boundary_list[0],'
                  'set as outline boundary')
        if 'type' in one_bound.keys():
            bound_type = one_bound['type']
        else:
            bound_type = 'open' # defaul type is open

        if 'h' in one_bound.keys():
            data_table.hSources[bound_ind] = np.array(one_bound['h'])
            bound_type = 'open'
        else:
            data_table.hSources[bound_ind] = None
        
        if 'hU' in one_bound.keys():
            data_table.hUSources[bound_ind] = np.array(one_bound['hU'])
            bound_type = 'open'
        else:
            data_table.hUSources[bound_ind] = None
        
        data_table.type[bound_ind] = bound_type
        if 'name' in one_bound.keys():
            data_table.name[bound_ind] = one_bound['name']

    return data_table

# private function called by Class Boundary
def _get_boundary_code(boudnary_data_table):
    """ Get the 3-element boundary code for h and hU
    boudnary_data_table: the boundary data table with
        columns ['type', 'hSources', 'hUSources']
    Return a new data_table added with h_code, hU_code, and description
    """
#        Get the three column boundary code
        #default outline boundary
    data_table = boudnary_data_table
    num_of_bound = data_table.shape[0]
    description = []
    n_seq = 0  # sequence of boundary
    m_h = 0  # sequence of boundary with IO source files
    m_hu = 0
    for n_seq in range(num_of_bound):
        
        data_table.h_code[n_seq] = np.array([[2, 0, 0]])
        bound_type = data_table.type[n_seq]
        if bound_type == 'rigid':
            data_table.hU_code[n_seq] = np.array([[2, 2, 0]])
            description1 = bound_type
        elif bound_type == 'fall':
            h_sources = np.array([[0, 0], [1, 0]])
            hU_sources = np.array([[0, 0, 0], [1, 0, 0]])
            data_table.h_code[n_seq] = np.array([[3, 0, m_h]])
            data_table.hU_code[n_seq] = np.array([[3, 0, m_hu]])
            m_h = m_h+1
            m_hu = m_hu+1
            description1 = bound_type+', h and hU fixed as zero'
        else: # open
            h_sources = data_table.hSources[n_seq]
            hU_sources = data_table.hUSources[n_seq]
            data_table.hU_code[n_seq] = np.array([[2, 1, 0]])
            description1 = bound_type
            if h_sources is not None:
                data_table.h_code[n_seq] = np.array([[3, 0, m_h]]) #[3 0 m]
                description1 = description1+', h given'
                m_h = m_h+1
            if hU_sources is not None:
                data_table.hU_code[n_seq] = np.array([[3, 0, m_hu]]) #[3 0 m]
                description1 = description1+', hU given'
                m_hu = m_hu+1
        description.append(description1)
    description[0] = '(outline) '+ description[0] # indicate outline boundary
    data_table['description'] = description
    return data_table

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

def _find_wet_io_cells(bound_obj, source_key='hUSources'):
    """ return subs of wet IO boundary cells
    """
    wet_cell_rows = []
    wet_cell_cols = []
    for ind_num in np.arange(bound_obj.num_of_bound):
        bound_source = bound_obj.data_table[source_key][ind_num]
        if bound_source is not None:
            source_value = np.unique(bound_source[:, 1:])
            # zero boundary conditions
            if not (source_value.size == 1 and source_value[0] == 0):
                cell_subs = bound_obj.cell_subs[ind_num]
                wet_cell_rows.append(cell_subs[0])
                wet_cell_cols.append(cell_subs[1])
    if len(wet_cell_rows)>0:
        wet_cell_rows = np.concatenate(wet_cell_rows, axis=0 )
        wet_cell_cols = np.concatenate(wet_cell_cols, axis=0 )
        cell_subs_wet_io = (wet_cell_rows, wet_cell_cols)
    else:
        cell_subs_wet_io = None           
    return cell_subs_wet_io

def _get_bound_normal(cell_subs, dem_obj):
    """get the angle of the normal vector of the bound line
    """
    cell_xy = sp.sub2map(cell_subs[0], cell_subs[1], dem_obj.header)
    cell_xy = np.c_[cell_xy[0], cell_xy[1]]
    if np.unique(cell_subs[1]).size==1: # vertical bound line
        theta = 0
    elif np.unique(cell_subs[0]).size==1: # horizontal bound line
        theta = np.pi*0.5
    else:
        boundary_slope = np.polyfit(cell_xy[:, 0], cell_xy[:, 1], 1)
        theta = np.arctan(boundary_slope[0])+np.pi/2
        if theta > np.pi*2:
            theta = theta-np.pi*2
        if theta < 0:
            theta = theta+np.pi*2
    start_xy = cell_xy[0]
    unit_vec = np.array([np.cos(theta), np.sin(theta)])
    end0_xy =  unit_vec*4*dem_obj.header['cellsize']+start_xy
    end1_xy = -unit_vec*4*dem_obj.header['cellsize']+start_xy
    end0_value = _get_array_value_by_rc(end0_xy, dem_obj)
    end1_value = _get_array_value_by_rc(end1_xy, dem_obj)
    if np.isnan(end0_value) & ~np.isnan(end1_value): # towards out
    # reverse direction of the boundline normal
        if theta >= np.pi:
            theta = theta-np.pi
        else:
            theta = theta+np.pi
    elif ~np.isnan(end0_value) & np.isnan(end1_value): # towards in
        pass
    else:
        warnings.warn('Cannot judge flow direction, please double check your boundary condition')
    return theta

def _get_array_value_by_rc(end_xy, dem_obj):
    rc_num = sp.map2sub(end_xy[0], end_xy[1], dem_obj.header)
    try:
        end_value = dem_obj.array[rc_num]
    except:
        end_value = np.nan
    if end_value == dem_obj.header['NODATA_value']:
        end_value == np.nan
    return end_value