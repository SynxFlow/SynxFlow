#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rainfall_processing
Created on Wed Dec  4 14:26:04 2019

@author: Xiaodong Ming
"""
import os
import warnings
import imageio
import shapefile # Requires the pyshp package
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
from .Raster import Raster
"""
Explaination of general parameters 
rain_source: (numpy array) rainfall source array
          The 1st column is usually time series in seconds, from the 2nd column
          towards end columns are rainfall rate in m/s
rain_mask: (numpy array) provide sequnce number of each gridded rainfall source
start_date: a datetime object to give the initial date and time of rain

"""
#%%
def get_time_series(rain_source, rain_mask=None, 
                     start_date=None, method='mean'):
    """ Plot time series of average rainfall rate inside the model domain   
    method: 'mean'|'max','min','mean'method to calculate gridded rainfall 
    over the model domain
    """
    if rain_mask is not None:
        rain_mask = rain_mask[~np.isnan(rain_mask)]
        rain_mask = rain_mask.astype('int32')
        rain_mask_unique = np.unique(rain_mask).flatten()
        rain_mask_unique = rain_mask_unique.astype('int32') 
        rain_source_valid = rain_source[:, rain_mask_unique+1]
    else:
        rain_source_valid = rain_source[:, 1:]
    time_series = rain_source[:,0]
    if type(start_date) is datetime:
        time_delta = np.array([timedelta(seconds=i) for i in time_series])
        time_x = start_date+time_delta
    else:
        time_x = time_series
    if method == 'mean':
        value_y = np.mean(rain_source_valid, axis=1)
    elif method == 'max':
        value_y = np.max(rain_source_valid, axis=1)
    elif method == 'min':
        value_y = np.min(rain_source_valid, axis=1)
    elif method == 'median':
        value_y = np.median(rain_source_valid, axis=1)
    elif method == 'sum':
        value_y = np.sum(rain_source_valid, axis=1)
    else:
        raise ValueError('Cannot recognise the calculation method')
    value_y =  value_y*3600*1000
    plot_data = np.c_[time_x, value_y]
    return plot_data

def get_spatial_map(rain_source, rain_mask_obj, figname=None,
                    method='sum', cellsize=None,
                    shp_file=None, dpi=200, title=None, **kwargs):
    """Get spatial rainfall map
    rain_mask_obj: asc file name or Raster object for rain mask
    cellsize: resample the rain_mask to a new grid (with larger cellsize)
    method: sum|mean caculate method for each cell, sum by time or mean by time 
    """
    # caculate rain source
    times = rain_source[:,0]
    rain_values = rain_source[:,1:]
    rain_total = np.trapz(rain_values, x=times, axis=0)*1000 #mm
    cax_str = 'mm/h'
    if method == 'sum':
        cell_rain = rain_total #mm
        cax_str = 'mm'
    elif method == 'mean':
        cell_rain = rain_total/(times.max()-times.min())*3600 #mm/h
    elif method == 'max':
        cell_rain = np.max(rain_values, axis=0)*1000*3600 #mm/h
    elif method== 'min':
        cell_rain = np.min(rain_values, axis=0)*1000*3600 #mm/h
    elif method== 'median':
        cell_rain = np.median(rain_values, axis=0)*1000*3600 #mm/h
    else:
        raise ValueError('Cannot recognise the calculation method')    
    # get spatial data
    if type(rain_mask_obj) is str:
        rain_mask_obj = Raster(rain_mask_obj)
    mask_obj = rain_mask_obj 
    if cellsize is not None:
        if cellsize > rain_mask_obj.header['cellsize']:
            mask_obj = rain_mask_obj.grid_resample_nearest(cellsize)  
    ind_nan = np.isnan(mask_obj.array)
    rain_mask = mask_obj.array+0
    rain_mask[ind_nan] = 0
    mask_ind = rain_mask.flatten(order='F').astype('int64')
    rain_vect = cell_rain[mask_ind]
    rain_array = np.reshape(rain_vect, mask_obj.array.shape, order='F')
    rain_array[ind_nan] = np.nan
    rain_map_obj = Raster(array=rain_array, header=mask_obj.header)
    fig, ax = rain_map_obj.mapshow(cax_str=cax_str, **kwargs)
    xbound = ax.get_xbound()
    ybound = ax.get_ybound()
    # draw shape file on the rainfall map
    if shp_file is not None:
        sf = shapefile.Reader(shp_file)
        for shape in sf.shapeRecords():
            x = [i[0] for i in shape.shape.points[:]]
            y = [i[1] for i in shape.shape.points[:]]
            ax.plot(x, y, color='r', linewidth=1)
    ax.set_xbound(xbound)
    ax.set_ybound(ybound)
    if title is not None:
        ax.set_title(title)
    if figname is not None:
        fig.savefig(figname, dpi=dpi, bbox_inches = 'tight', pad_inches = 0.02)
        plt.close(fig)
    return rain_map_obj

def plot_time_series(plot_data=None, rain_source=None, rain_mask=None,
                     method='mean', start_date=None, datetime_interval=24, 
                     datetime_format='%m-%d', title_str=None,
                     **kwargs):
    """ Plot time series of average rainfall rate inside the model domain   
    method: 'mean'|'max','min','mean'method to calculate gridded rainfall 
    over the model domain
    """
    if plot_data is None:
        plot_data = get_time_series(rain_source, rain_mask, start_date, method)
    time_x = plot_data[:,0]
    value_y = plot_data[:,1]
    fig, ax = plt.subplots()
    ax.plot(time_x, value_y, **kwargs)
    if 'start_date' in kwargs.keys():
        ax.xaxis.set_major_locator(mdates.HourLocator(
                interval=datetime_interval))
        ax.xaxis.set_major_formatter(mdates.DateFormatter(datetime_format))
    ax.set_ylabel(method+' rainfall rate (mm/h)')
    ax.grid(True)
    if title_str is None:
        title_str = method+' precipitation in the model domain'
    ax.set_title(title_str)
    plt.show()
    return fig, ax

def create_animation(output_file, rain_source, mask_file,
                     duration=0.5, **kwargs):
    """ Create animation of gridded rainfall rate    
    mask_header: (dict) header file provide georeference of rainfall mask
    start_date: a datetime object to give the initial date and time of rain
    duration: duration for each frame (seconds)
    cellsize: sclar (meter) the size of rainfall grid cells
    """
    fig_names = create_pictures(rain_source, mask_file, **kwargs)
    # create animation with the images
    images = []
    for fig_name in fig_names:
        images.append(imageio.imread(fig_name))
        os.remove(fig_name)
    # save animation and delete images
    if not output_file.endswith('.gif'):
        output_file = output_file+'.gif'
    imageio.mimsave(output_file, images, duration=duration)

def create_mp4(output_file, rain_source, mask_file, fps=10, **kwargs):
    fig_names = create_pictures(rain_source, mask_file, **kwargs)    
    if not output_file.endswith('.mp4'):
        output_file = output_file+'.mp4'
    print(output_file)
    writer = imageio.get_writer(output_file, 'MP4', fps=fps)
    for fig_name in fig_names:
        writer.append_data(imageio.imread(fig_name))
        os.remove(fig_name)
    writer.close()
    
def create_pictures(rain_source, mask_file, cellsize=1000, 
                    start_date=None, dpi=100, shp_file=None, **kwargs):
    """ create rainfall rate images
    rain_source
    mask_file: a arc grid or a Raster object
    """
    time_series = rain_source[:,0]
    # create images
    fig_names = []
    for i in np.arange(0, time_series.size):
        print(str(i)+'/'+str(time_series.size))
        fig_name = 'temp'+str(i)+'.png'
        fig_names.append(fig_name)
        one_source = rain_source[i, :]
        one_source = one_source.reshape((1, one_source.size))
        if start_date is None:
            title_str = '{:.0f}'.format(time_series[i])+'s'
        else:
            title_str = start_date+timedelta(seconds=time_series[i])
            title_str = title_str.strftime("%m/%d/%Y %H:%M:%S")
        get_spatial_map(one_source, mask_file, cellsize=cellsize,
                        shp_file=shp_file, title=title_str,
                        figname=fig_name, dpi=dpi, **kwargs)
    return fig_names
    
def _check_rainfall_rate_values(rain_source, times_in_1st_col=True):
    """ Check the rainfall rate values in rain source array
    times_in_1st_col: indicate whether the first column is times
    Return:
        values_max: maximum rainfall rate in mm/h
        values_mean: average rainfall rate in mm/h
    """
    # get the pure rainfall rate values
    if times_in_1st_col:
        rain_values = rain_source[:, 1:]
        time_series = rain_source[:, 0]
    else:
        rain_values = rain_source
        time_series = np.arange(rain_values.shape[0])
    # convert the unit of rain rate values from m/s to mm/h
    rain_values_mmh = rain_values*3600*1000
    values_max = rain_values_mmh.max()
    values_mean = rain_values.mean(axis=1)
    rain_total_amount = np.trapz(y=values_mean, x=time_series) # in meter
    duration = np.ptp(time_series)
    rain_rate_mean = rain_total_amount*1000/(duration/3600) #mm/h
    if values_max > 1000 or rain_rate_mean > 500:
        warnings.warn('Very large rainfall rates, better check your data!')
        print('Max rain: {:.2f} mm/h, Average rain: {:.2f} mm/h'.\
              format(values_max, rain_rate_mean))
    return values_max, rain_rate_mean