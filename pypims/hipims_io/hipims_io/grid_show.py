#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_show.py
To do:
    To visulize grid data, e.g. raster objec(s)
    static map functions:
    1. mapshow: general map of the grid values
    2. rankshow: show grid values in ranks
    3. hillshade: show a hillshade map of a grid
    4. vectorshow: show a vector map of two grids       
    animation functions:
    5. make_gif: create a gif file to show values of a series of grids
    6. make_mp4: create a video file to show values of a series of grids
Created on Tue Mar 10 15:37:28 2020

@author: Xiaodong Ming
"""
import os
import copy
import imageio
import numpy as np
import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import matplotlib.colors as colors
from . import spatial_analysis as sp
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable
#%% draw inundation map with domain outline
def mapshow(raster_obj=None, array=None, header=None, ax=None,
            figname=None, figsize=None, dpi=300, title=None,
            cax=True, cax_str=None, relocate=False, scale_ratio=1, **kwargs):
    """
    Display raster data without projection
    raster_obj: a Raster object
    array, header: to make Raster object if raster_obj is not given
    figname: the file name to export map, if figname is empty, then
        the figure will not be saved
    figsize: the size of map
    dpi: The resolution in dots per inch
    vmin and vmax define the data range that the colormap covers
    **kwargs: keywords argument of function imshow
    """
    if raster_obj is not None:
        array = raster_obj.array
        header = raster_obj.header
    # change NODATA_value to nan
    np.warnings.filterwarnings('ignore')
    array = array+0
    ind = array == header['NODATA_value']
    if ind.sum()>0:
        array = array.astype('float32')
        array[ind] = np.nan
    # adjust tick label and axis label
    map_extent = sp.header2extent(header)    
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = ax.get_figure()
    img = ax.imshow(array, extent=map_extent, **kwargs)
    _adjust_axis_tick(ax, relocate, scale_ratio)
    # add colorbar
	# create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    if cax==True:
        _ = _set_continous_colorbar(ax, img, cax_str)
    ax.axes.grid(linestyle='-.', linewidth=0.2)
    ax.set_aspect('equal', 'box')
    if title is not None:
        ax.set_title(title)
    # save figure
    if figname is not None:
        fig.savefig(figname, dpi=dpi, bbox_inches='tight', pad_inches=0)
    # return fig and axis handles
    return fig, ax

def rankshow(raster_obj=None, array=None, header=None, figname=None, 
             figsize=None, dpi=200, ax=None, color='Blues',
             breaks=[0.2, 0.3, 0.5, 1, 2],
             colorbar_kw=None, legend_kw=None,
             relocate=False, scale_ratio=1, alpha=1, **kwargs):
    """ Display water depth map in ranks defined by breaks
    breaks: list of values to define rank. Array values lower than the first
        break value are set as nodata.
    color: color series of the ranks
    colorbar_kw: dict, keyword arguments to set colorbar
    legend_kw: dict, keyword arguments to set legend
    
    """
    if raster_obj is not None:
        array = raster_obj.array
        header = raster_obj.header
    np.warnings.filterwarnings('ignore')
    ind = array == header['NODATA_value']
    array = array.astype('float32')+0
    array[ind] = np.nan
    # create color ranks
    array, newcmp, norm= _set_color_rank(array, breaks, color)
    map_extent = sp.header2extent(header)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    chm_plot = ax.imshow(array, extent=map_extent, 
                         cmap=newcmp, norm=norm, alpha=alpha) 
    _adjust_axis_tick(ax, relocate, scale_ratio)
    # create colorbar
    if colorbar_kw is not None:
        _set_rank_colorbar(ax, chm_plot, norm, colorbar_kw)
    if legend_kw is not None: # legend
        _set_color_legend(ax, norm, newcmp, legend_kw)
    if figname is not None:
        fig.savefig(figname, dpi=dpi)
    return fig, ax

def hillshade(raster_obj, figsize=None, azdeg=315, altdeg=45, vert_exag=1,
              cmap=None, blend_mode='overlay', alpha=1, relocate=False,
              scale_ratio=1,):
    """ Draw a hillshade map
    """
    array = raster_obj.array+0
    array[np.isnan(array)] = np.nanmax(array)
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    if cmap is None:
        cmap = plt.cm.gist_earth
    else:
        cmap = plt.get_cmap(cmap)
    fig, ax = plt.subplots(figsize=figsize)
    rgb = ls.shade(array, cmap=cmap, blend_mode=blend_mode, 
                   vert_exag=vert_exag)
    ax.imshow(rgb, extent=raster_obj.extent, alpha=alpha)
    _adjust_axis_tick(ax, relocate, scale_ratio)
    return fig, ax

def vectorshow(obj_x, obj_y, figname=None, figsize=None, dpi=300, **kwargs):
    """
    plot velocity map of U and V, whose values stored in two raster
    objects seperately
    """
    X, Y = obj_x.to_points()        
    U = obj_x.array
    V = obj_y.array
    if U.shape!=V.shape:
        raise TypeError('bad argument: the shapes must be the same')
    if 'figsize' in kwargs:
        figsize = kwargs['figsize']
    else:
        figsize = None
    fig, ax = plt.subplots(1, figsize=figsize)
    plt.quiver(X, Y, U, V)
    ax.set_aspect('equal', 'box')
    ax.tick_params(axis='y', labelrotation=90)
    if figname is not None:
        fig.savefig(figname, dpi=dpi)
    return fig, ax

def make_gif(output_file, obj_list=None, header=None, array_3d=None, 
                     time_str=None, breaks=None, fig_names=None,
                     duration=0.5, delete=False, **kwargs):
    """ Create animation of gridded data    
    mask_header: (dict) header file provide georeference of rainfall mask
    start_date: a datetime object to give the initial date and time of rain
    duration: duration for each frame (seconds)
    cellsize: sclar (meter) the size of rainfall grid cells
    """
    if fig_names is None:
        fig_names = _plot_temp_figs(obj_list, header, array_3d, breaks,
                                time_str, **kwargs)
    images = []
    for fig_name in fig_names:
        images.append(imageio.imread(fig_name))
        if delete:
            os.remove(fig_name)
    # save animation and delete images
    if not output_file.endswith('.gif'):
        output_file = output_file+'.gif'
    imageio.mimsave(output_file, images, duration=duration)

def make_mp4(output_file, obj_list=None, header=None, array_3d=None, 
               time_str=None, breaks=None, fig_names=None, delete=False,
               fps=10, **kwargs):
    """ Create a video file based on a series of grids
    obj_list: a list of Raster objects
    header: a header dict providing georeference the grid [not necessary if 
                                                           obj_list was given]
    array_3d: a 3D numpy array storing grid values for each timestep 
                (in 1st dimension), [not necessary if obj_list was given]
    time_str: a list of string to show time information for each frame
    """
    if fig_names is None:
        fig_names = _plot_temp_figs(obj_list, header, array_3d, breaks,
                                    time_str, **kwargs)
    if not output_file.endswith('.mp4'):
        output_file = output_file+'.mp4'
    print(output_file)
    writer = imageio.get_writer(output_file, 'MP4', fps=fps)
    for fig_name in fig_names:
        writer.append_data(imageio.imread(fig_name))
        if delete:
            os.remove(fig_name)
    writer.close()

def plot_shape_file(shp_file, figsize=None, ax=None, color='r', linewidth=0.5,
                    **kw):
    """plot a shape file to a map axis
    """
    import shapefile
    sf = shapefile.Reader(shp_file)
    if ax is None:
        fig, ax = plt.subplots(1, figsize=figsize)
#        xbound = None
#        ybound = None
    else:
        fig = ax.get_figure()
#        xbound = ax.get_xbound()
#        ybound = ax.get_ybound()
    # draw shape file on the rainfall map
    for shape in sf.shapeRecords():
        for i in range(len(shape.shape.parts)):
            i_start = shape.shape.parts[i]
            if i==len(shape.shape.parts)-1:
                i_end = len(shape.shape.points)
            else:
                i_end = shape.shape.parts[i+1]
            x = [i[0] for i in shape.shape.points[i_start:i_end]]
            y = [i[1] for i in shape.shape.points[i_start:i_end]]
            ax.plot(x, y, color=color, linewidth=linewidth, **kw)
#    ax.set_xbound(xbound)
#    ax.set_ybound(ybound)
    return fig, ax

def _plot_temp_figs(obj_list=None, header=None, array_3d=None,
                    breaks=None, time_str=None, **kwargs):
    """plot a series of temp pictures and save to make animation
    plot_fun: the function to plot
    """
    """ Create a video file based on a series of grids
    """
    if obj_list is not None:
        header = obj_list[0].header
        array_3d = []
        for grid_obj in obj_list:
            array_3d.append(grid_obj.array)
        array_3d = np.array(array_3d)
    if breaks is None: #plot continous map
        plot_fun = mapshow
    else:
        plot_fun = rankshow
        kwargs['breaks']=breaks
    fig_names = []
    for i in np.arange(array_3d.shape[0]):
        fig_name = 'temp'+str(i)+'.png'
        fig, ax = plot_fun(array=array_3d[i],header=header, **kwargs)
        if type(time_str) is list:
            ax.set_title(time_str[i])
        fig.savefig(fig_name)
        plt.close(fig)
        fig_names.append(fig_name)
    return fig_names
    
#%%
def _set_color_rank(array, breaks, color):
    """Set color rank for array values 
    """
    array[array < breaks[0]] = np.nan
    max_value = np.nanmax(array)
    breaks = copy.deepcopy(breaks)
    if breaks[-1] < max_value:
        breaks.append(max_value)   
    norm = colors.BoundaryNorm(breaks, len(breaks))
    newcolors = cm.get_cmap(color, norm.N)
    newcolors = newcolors(np.linspace(0, 1, norm.N))
    newcmp = ListedColormap(newcolors)     
    return array, newcmp, norm

def _set_rank_colorbar(ax, img, norm):
    """ Set color bar for rankshow on the right of the ax
    """    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img, cax=cax)
    y_tick_values = cax.get_yticks()
    boundary_means = [np.mean((y_tick_values[ii],y_tick_values[ii-1])) 
                        for ii in range(1, len(y_tick_values))]
    print(norm.boundaries)
    category_names = [(str(norm.boundaries[ii-1])+'~'+
                       str(norm.boundaries[ii]))
                      for ii in range(1, len(norm.boundaries))]
#    category_names[0] = '<='+str(norm.boundaries[1])
    category_names[-1] = '>'+str(norm.boundaries[-2])
    cax.yaxis.set_ticks(boundary_means)
    cax.yaxis.set_ticklabels(category_names,rotation=0)
    return cax

def _set_color_legend(ax, norm, cmp, legend_kw):
    """ Set color legend attributes
    legend_kw: dict, keyword arguments to set legend, eg:
        {loc:'lower right', bbox_to_anchor:(1,0), facecolor:None}
    """
    category_names = [(str(norm.boundaries[ii-1])+'~'+
                       str(norm.boundaries[ii]))
                      for ii in range(1, len(norm.boundaries))]
#    category_names[0] = '<='+str(norm.boundaries[1])
    category_names[-1] = '>'+str(norm.boundaries[-2])
    ii = 0
    legend_labels = {}
    for category_name in category_names:
        legend_labels[category_name] = cmp.colors[ii,]
        ii = ii+1
    patches = [Patch(color=color, label=label)
               for label, color in legend_labels.items()]
    ax.legend(handles=patches, **legend_kw)
    return ax

def _set_continous_colorbar(ax, img, cax_str=None,
                            loc='right', size='3%', pad=0.05,
                            fontsize='small'):
    """ Set a continous color bar
    cbar = _set_continous_colorbar(ax, img, cax_str=None)
    cax_str: title of color bar
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(loc, size=size, pad=pad)
    cbar = plt.colorbar(img, cax=cax)
    if cax_str is not None:
        cbar.ax.set_xlabel(cax_str, horizontalalignment='left',
                           fontsize=fontsize)
        cbar.ax.xaxis.set_label_coords(0, 1.06)
    return cbar

def _adjust_axis_tick(ax, relocate=True, scale_ratio=1):
    """
    Adjust the axis tick to a new staring point and/or new unit 
    Example:
        if scale_ratio = 1000, and the original extent unit is meter,
        then the unit is converted to km, and the extent is divided by 1000
    """
    xticks = ax.get_xticks()
    x_space = xticks[1]-xticks[0]
    yticks = ax.get_yticks()
    y_space = yticks[1]-yticks[0]
    if relocate:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xticks = np.arange(xlim[0], xlim[1], x_space)
        xticks_label = xticks-xlim[0]
        yticks = np.arange(ylim[0], ylim[1], y_space)
        yticks_label = yticks-ylim[0]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
    else:
        xticks_label = xticks
        yticks_label = yticks
    if scale_ratio == 1000:
        label_tag = 'km'
        xticks_label = (xticks_label/1000).astype('int64')
        yticks_label = (yticks_label/1000).astype('int64')
    else:
        label_tag = 'meter'
        xticks_label = (xticks_label).astype('int64')
        yticks_label = (yticks_label).astype('int64')
    ax.set_xticklabels(xticks_label)
    ax.set_yticklabels(yticks_label)
    ax.set_xlabel(label_tag+' towards east')
    ax.set_ylabel(label_tag+' towards north')
    return None

def main():
    print('Package to show grid data')

if __name__=='__main__':
    main()