#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Tue Mar 31 16:20:13 2020
# Author: Xiaodong Ming

"""
Raster
=======

To do:
    1. Read, write, analyze, and visualise gridded raster data
    2. Convert shape data to raster data

---------------

"""
import os
import copy
import math
import fiona
import numpy as np
import rasterio as rio
from scipy import interpolate
from . import spatial_analysis as sp
from . import grid_show as gs
#%% *******************************To deal with raster data********************
#   ***************************************************************************    
class Raster(object):
    """    
    To deal with raster data with a ESRI ASCII or GTiff format

    Attributes:
        source_file: file name to read grid data 
        output_file: file name to write a raster object 
        array: a numpy array storing grid cell values 
        header: a dict storing reference information of the grid, with keys:
            nrows, ncols, xllcorner, yllcorner, cellsize, NODATA_value
        extent: a tuple storing outline limits of the raster (left, right, 
            bottom, top)
        shape: shape of the Raster value array
        cellsize: the length of each square cell
        extent_dict: a dictionary storing outline limits of the raster  
        wkt: (string) the Well-Known_Text (wkt) projection information  
        
    """
#%%======================== initialization function ===========================   
    def __init__(self, source_file=None, array=None, header=None, 
                 xllcorner=0, yllcorner=0, cellsize=100, NODATA_value=-9999,
                 crs=None, num_header_rows=6):
        """Initialise the object

        Args:
            source_file: name of a asc/tif file if a file read is needed
            array: values in each raster cell [a numpy array]
            header: georeference of the raster [a dictionary containing 6 keys]
                nrows, nclos [int]
                cellsize, xllcorner, yllcorner
                NODATA_value
            crs: coordinate reference system, either epsg(epsg code, int) or
                wkt(string), or a rasterio.crs object
        Example: define a raster object with a random array
            array = np.random.rand(10, 10)
            header = {'ncols':array.shape[1], 'nrows':array.shape[0],
                      'xllcorner':0, 'yllcorner':1,
                      'cellsize':100, 'NODATA_value':-9999}
            obj_ras = Raster(array=array, header=header)
            obj_ras.mapshow() %plot map

        """
        # get data from file or arguments
        if type(source_file) is str: # tif, txt, asc, or gz file
            if source_file.endswith('.tif'): # only read the first band
                array, header, crs = sp.tif_read(source_file)
            else:
                array, header, crs = sp.arcgridread(source_file,
                                                    num_header_rows)                
            self.source_file = source_file
        elif type(source_file) is bytes:  # try a binary file-like object
            array, header = sp.byte_file_read(source_file)
            self.source_file = None
        
        # projection
        if crs is not None:
            self.set_crs(crs)
        # header and extent
        if header is None:
            header = {'ncols':array.shape[1], 'nrows':array.shape[0],
                      'xllcorner':xllcorner, 'yllcorner':yllcorner,
                      'cellsize':cellsize, 'NODATA_value':NODATA_value}
        self.header = header
        self.extent = sp.header2extent(header)
        self.extent_dict = {'left':self.extent[0], 'right':self.extent[1],
                            'bottom':self.extent[2], 'top':self.extent[3]}
        
        # raster value array
        
        ind = array == header['NODATA_value']
        if ind.sum() > 0:
            self.array = array+0.0
            self.array[ind] = np.nan
        else:
            self.array = array
            
        
        # shape and cellsize
        self.shape = self.array.shape
        if array.shape != (header['nrows'], header['ncols']):
            raise ValueError(('shape of array is not consistent with '
                              'nrows and ncols in header'))
        self.cellsize = header['cellsize']
        
        # num_valid_cells
        self.num_valid_cells = np.sum(np.isnan(self.array))
        self.size = self.array.size
        
        # summary
        self.get_summary()
        
        self.to_rasterio = self.write_tif

            
#%%============================= Spatial analyst ==============================   
    def get_summary(self):
        """Get information summary of the object

        Returns
        -------
        summary : dict
            information summary of the object.

        """
        summary = copy.deepcopy(self.header)
        summary['num_valid_cells'] = self.num_valid_cells
        if hasattr(self, 'crs'):
            summary['crs'] = self.crs.data
        if hasattr(self, 'source_file'):
            summary['source_file'] = self.source_file
        self._summary = summary
        return summary

    def set_crs(self, crs):
        """set reference coordinate system
        Parameters
        ----------
        crs : int|string|rasterio.crs object
            epsg code|wkt|asterio.crs object
        """
        if type(crs) is str:
            self.crs = rio.crs.CRS.from_wkt(crs)
        elif type(crs) is int:
            self.crs = rio.crs.CRS.from_epsg(crs)
        elif type(crs) is rio.crs.CRS:
            self.crs = crs
        else:
            raise IOError('crs must be int|string|rasterio.crs object')

    def rect_clip(self, clip_extent):
        """clip raster according to a rectangle extent

        Args:
            clip_extent: list of [left, right, bottom, top]

        Return:
            Raster: a new raster object
        """
        X = clip_extent[0:2]
        Y = clip_extent[2:4]
        rows, cols = sp.map2sub(X, Y, self.header)
        x_centre, y_centre = sp.sub2map(rows, cols, self.header)
        xllcorner = min(x_centre)-0.5*self.header['cellsize']
        yllcorner = min(y_centre)-0.5*self.header['cellsize']
        header_new = copy.deepcopy(self.header)
        if max(rows) >= self.header['nrows']:
            max_row = self.header['nrows']
        else:
            max_row = max(rows)+1
        if max(cols) >= self.header['ncols']:
            max_col = self.header['ncols']
        else:
            max_col = max(cols)+1
        if min(rows) <= 0:
            min_row = 0
        else:
            min_row = min(rows) 
        if min(cols) <= 0:
            min_col = 0
        else:
            min_col = min(cols) 
        array_new = self.array[min_row:max_row,
                               min_col:max_col]
        header_new['nrows'] = array_new.shape[0]
        header_new['ncols'] = array_new.shape[1]
        header_new['xllcorner'] = xllcorner
        header_new['yllcorner'] = yllcorner
        obj_new = Raster(array=array_new, header=header_new)
        if hasattr(self, 'crs'):
            obj_new.crs = self.crs
        return obj_new
    
    def clip(self, clip_mask=None):
        """clip raster according to a mask

        Args:
            mask: 1. string name of a shapefile or 2. 2-col numpy array giving
                X and Y coords in each column to shape the mask polygon
        
        Return:
            Raster: a new raster object
        """
        from rasterio import mask
        ds_rio = self.to_rasterio_ds()
        if type(clip_mask) is str:
            with fiona.open(clip_mask, 'r') as shapefile:
                shapes = [feature['geometry'] for feature in shapefile]
        elif type(clip_mask) is np.ndarray:
            shape = {'type':'Polygon', 'coordinates':[clip_mask]}
            shapes = [shape]
        else:
            raise IOError('mask must be either a string or a numpy array')
        
        out_image, out_transform = mask.mask(ds_rio, shapes, crop=True) #
        ds_rio.close()
        array_new = out_image[0]
        cellsize = out_transform[0]
        xllcorner = out_transform[2]
        yllcorner = out_transform[5]- cellsize*array_new.shape[0]
        header_new = copy.deepcopy(self.header)
        header_new['nrows'] = array_new.shape[0]
        header_new['ncols'] = array_new.shape[1]
        header_new['xllcorner'] = xllcorner
        header_new['yllcorner'] = yllcorner
        obj_new = Raster(array=array_new, header=header_new)
        if hasattr(self, 'crs'):
            obj_new.crs = self.crs
        return obj_new
    
    def rasterize(self, shp_filename):
        """
        rasterize the shapefile to the raster object and return a bool array
            with Ture value in and on the polygon/polyline

        Args:
            shp_filename: string for shapefile name, or a list of shapefile
                geometry with attributes 'type' and 'coordinates'

        Return:
            index_array: an array of True|False values masking the shapefile

        """
        from rasterio import mask
        ds_rio = self.to_rasterio_ds()
        if type(shp_filename) is str:
            with fiona.open(shp_filename, 'r') as shapefile:
                shapes = [feature['geometry'] for feature in shapefile]
        else:
            shapes = shp_filename
        shapes = [x for x in shapes if x != None]
        out_image, _ = mask.mask(ds_rio, shapes) #, crop=True
        rasterized_array = out_image[0]
        rasterized_array[np.isnan(rasterized_array)] = ds_rio.nodata
        index_array = np.full(rasterized_array.shape, True)
        index_array[rasterized_array == ds_rio.nodata] = False
        ds_rio.close()
        return index_array
        
    def resample(self, new_cellsize, method='bilinear'):
        """ Resample the raster object to a new resolution
        
        Args:
            new_cellsize: scalar, the resoltion of the new raster object
            method: string, one of the values including 'nearest', 'bilinear',
            'cubic', 'cubic_spline', 'lanczos', 'average', 'mode', 'gauss',
            'max', 'min', 'med', 'q1', 'q3' 
        
        Return:
            Raster object
        """
        from rasterio.enums import Resampling
        method_list = ['nearest', 'bilinear', 'cubic', 'cubic_spline',
                       'lanczos', 'average', 'mode', 'gauss', 'max', 'min',
                       'med', 'q1', 'q3']
        ind = method_list.index(method)
        upscale_factor = self.cellsize/new_cellsize
        ds_rio = self.to_rasterio_ds()
        new_shape = (1, int(ds_rio.height * upscale_factor),
                        int(ds_rio.width * upscale_factor))
        resampling_method = Resampling(ind)
        data = ds_rio.read(out_shape=new_shape, resampling=resampling_method)
        data = data[0]
        # scale image transform
        transform = ds_rio.transform * ds_rio.transform.scale(
                    (ds_rio.width / data.shape[-1]),
                    (ds_rio.height / data.shape[-2]))
        ds_rio.close()
        new_header = copy.deepcopy(self.header)
        new_header['cellsize'] = new_cellsize
        new_header['nrows'] = data.shape[0]
        new_header['ncols'] = data.shape[1]
        new_header['xllcorner'] = transform[2]
        new_header['yllcorner'] = transform[5]-data.shape[0]*new_cellsize
        obj_new = Raster(array=data, header=new_header)
        if hasattr(self, 'crs'):
            obj_new.crs = self.crs
        return obj_new
        
    
    def point_interpolate(self, points, values, method='nearest'):
        """ Interpolate values of 2D points to all cells on the Raster object

        2D interpolate

        Args:
            points: ndarray of floats, shape (n, 2)
                Data point coordinates. Can either be an array of shape (n, 2), 
                or a tuple of ndim arrays.
            values: ndarray of float or complex, shape (n, )
                Data values.
            method: {'linear', 'nearest', 'cubic'}, optional
                Method of interpolation.

        """
        grid_x, grid_y = self.to_points()
        array_interp = interpolate.griddata(points, values, (grid_x, grid_y),
                                            method=method)
        obj_new = copy.deepcopy(self)
        obj_new.array = array_interp
        return obj_new
    
    def grid_interpolate(self, value_grid, method='nearest'):
        """ Interpolate values of a grid to all cells on the Raster object

        2D interpolate

        Args:
            value_grid: a grid file string or Raster object 
            method: {'linear', 'nearest', 'cubic'}, optional
                Method of interpolation.

        Return: 
            numpy array: the interpolated grid with the same size of the self 
                object
        """
        if type(value_grid) is str:
            value_grid = Raster(value_grid)
        points_x, points_y = value_grid.to_points()
        points = np.c_[points_x.flatten(), points_y.flatten()]
        values = value_grid.array.flatten()
        ind_nan = ~np.isnan(values)
        grid_x, grid_y = self.to_points()
        array_interp = interpolate.griddata(points[ind_nan, :],
                                            values[ind_nan], (grid_x, grid_y),
                                            method=method)
        return array_interp
    
    def grid_resample_nearest(self, newsize):
        """resample a grid to a new grid resolution via nearest interpolation

        Alias: GridResample
        """
        if isinstance(newsize, dict):
            header = newsize.copy()
        else:
            oldsize = self.header['cellsize']
            header = copy.deepcopy(self.header)
            header['cellsize'] = newsize
            ncols = math.floor(oldsize*self.header['ncols']/newsize)
            nrows = math.floor(oldsize*self.header['nrows']/newsize)
            header['ncols'] = ncols
            header['nrows'] = nrows
        #centre of the first cell in array
        x11 = header['xllcorner']+0.5*header['cellsize']
        y11 = header['yllcorner']+(header['nrows']-0.5)*header['cellsize']
        x_all = np.linspace(x11, x11+(header['ncols']-1)*header['cellsize'],
                            header['ncols'])
        y_all = np.linspace(y11, y11-(header['nrows']-1)*header['cellsize'],
                            header['nrows'])
        row_all, col_all = sp.map2sub(x_all, y_all, self.header)
        rows, cols = np.meshgrid(row_all, col_all) # nrows*ncols array
        array = self.array[rows, cols]
        array = array.transpose()
        array = array.astype(self.array.dtype)
        obj_new = Raster(array=array, header=header)
        if hasattr(self, 'crs'):
            obj_new.crs = self.crs
        return obj_new
    
    def assign_to(self, new_header):
        """ Assign_to the object to a new grid defined by new_header 

        If their cellsizes are not equal, the original Raster will be 
        resampled to the target grid.

        Return:
            Raster: A newly defined grid
        """
        rows = np.arange(0, new_header['nrows'])
        cols = np.arange(0, new_header['ncols'])
        X, Y = sp.sub2map(rows, cols, new_header)
        grid_x, grid_y = np.meshgrid(X, Y)
        rows, cols = sp.map2sub(grid_x, grid_y, self.header)
        rows[rows > self.header['nrows']-1] = self.header['nrows']-1
        rows[rows < 0] = 0
        cols[cols > self.header['ncols']-1] = self.header['ncols']-1
        cols[cols < 0] = 0
        new_array = self.array[rows, cols]
        new_array = new_array+0.0
        new_array[new_array == self.header['NODATA_value']] = np.nan
        obj_new = Raster(array=new_array, header=new_header)
        if hasattr(self, 'crs'):
            obj_new.crs = self.crs
        return obj_new
    
    def paste_on(self, obj_large, ignore_nan=True):
        """ Paste the object to a larger grid defined by obj_large and
        replace corresponding grid values with the object array

        If their cellsizes MUST be equal
        """
        header_s = self.header
        header_l = obj_large.header
        x, y = self.to_points()
        r0, c0 = sp.map2sub(self.extent[0]+self.cellsize/2, 
                            self.extent[-1]-self.cellsize/2,
                            header_l)
        rows = np.arange(r0, r0+header_s['nrows'])
        cols = np.arange(c0, c0+header_s['ncols'])
        # cut array outside the range of the large raster grid
        ind_r = np.logical_and(rows > 0, rows <= header_l['nrows']-1)
        rows = rows[ind_r]
        ind_c = np.logical_and(cols > 0, cols <= header_l['ncols']-1)
        cols = cols[ind_c]
        array_small = self.array[ind_r, :]
        array_small = array_small[:, ind_c]
        rows_grid, cols_grid = np.meshgrid(rows, cols, indexing='ij')
        if ignore_nan:
            array_large = obj_large.array[rows_grid, cols_grid]
            ind_nan = np.isnan(array_small)
            array_small[ind_nan] = array_large[ind_nan]
        obj_large.array[rows_grid, cols_grid] = array_small
        return obj_large

    def to_points(self):
        """ Get X and Y coordinates of all raster cells

        Return:
            numpy array:  coordinates of the raster object
        """
        ny, nx = self.array.shape
        cellsize = self.header['cellsize']
        # coordinate of the centre on the top-left pixel
        x00centre = self.extent_dict['left'] + cellsize/2
        y00centre = self.extent_dict['top'] - cellsize/2
        x = np.arange(x00centre, x00centre+cellsize*nx, cellsize)
        y = np.arange(y00centre, y00centre-cellsize*ny, -cellsize)
        xv, yv = np.meshgrid(x, y)
        return xv, yv
    
    def write_asc(self, output_file, compression=False, export_prj=False):
        """ write raster as asc format file

        Alias: Write_asc

        Args:
            output_file: output file name
            compression: logic, whether compress write the asc file as gz
        """
        sp.arcgridwrite(output_file, self.array, self.header, compression)
        # if projection is defined, write .prj file for asc file
        if export_prj & hasattr(self, 'crs'):
            prj_file = output_file[0:-4]+'.prj'
            wkt = self.crs.to_wkt()
            with open(prj_file, "w") as prj:        
                prj.write(wkt)
        return None
    
    def write(self, output_file, compression=False):
        """ Export to a file, tif, asc, txt, or gz
        Parameters
        ----------
        output_file : string
            file name, ends with tif, asc, txt, or gz.
        compression : logical, optional
            whether compress or not. The default is False.

        Returns
        -------
        None.

        """

        if output_file.endswith('.gz'):
            self.write_asc(output_file, compression=True)
        elif output_file.endswith('.tif'):
            self.write_tif(output_file)
        else:
            self.write_asc(output_file, compression)
        
    
    def write_tif(self, output_file, src_epsg=27700):
        """ Convert to a rasterio dataset
        
        Args:
            output_file: a string to give output file name
            src_epsg: int scalar to give EPSG code of the coordinate reference
                system of the original dataset, default is 27700 for BNG

        """
        
        if output_file.endswith('.tif'):
            filename = output_file
        else:
            filename = output_file+'.tif'
        if not hasattr(self, 'meta'):
            self.get_meta(src_epsg)
        meta = self.meta # dictionary
        array_data = self.array+0
        nomask = np.isnan(array_data)
        array_data[nomask] = meta['nodata']
        with rio.open(filename, 'w', **meta) as out_f:
            out_f.write(array_data, 1)
    
    def to_rasterio_ds(self):
        
        if not hasattr(self, 'meta'):
            self.get_meta()
        meta = self.meta
        filename = 'temp.tif'
        ds_rio = rio.open(filename, 'w+', **meta)
        ds_rio.write(self.array, 1)
        os.remove(filename)
        return ds_rio

    
    def get_meta(self, src_epsg=27700):
        """ Get rasterio meta data
        """
        from rasterio.transform import Affine
        dx = self.cellsize
        x = self.extent_dict['left'] # upper-left corner of the first pixel
        y = self.extent_dict['top']
        transform = Affine.translation(x, y)*Affine.scale(dx, -dx)
        if not hasattr(self, 'crs'):
            crs = rio.crs.CRS.from_epsg(src_epsg)
        else:
            crs = self.crs
        ras_meta = {'driver': 'GTiff',
                    'dtype': self.array.dtype.name,
                    'nodata': self.header['NODATA_value'],
                    'width': self.array.shape[1],
                    'height': self.array.shape[0],
                    'count': 1,
                    'crs': crs,
                    'transform': transform}
        self.meta = ras_meta
        
    
    def reproject(self, dst_epsg, output_file=None):
        """Reproject the raster to a different coordinate referenece system
        
        Args:
            output_file: a string to give output file name
            src_epsg: int scalar to give EPSG code of the coordinate reference
                system of the original dataset, default is 27700 for BNG
        Return:
            dst_rio: destination rasterio dataset
        
        """
        import rasterio
        from rasterio.warp import calculate_default_transform as cal_tsf
        from rasterio.warp import reproject, Resampling
        src_rio = self.to_rasterio_ds()
        dst_crs = rasterio.crs.CRS.from_epsg(dst_epsg)
        transform, width, height = cal_tsf(src_rio.crs, dst_crs, src_rio.width, 
                                           src_rio.height, *src_rio.bounds)
        kwargs = src_rio.meta.copy()
        kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width,
                       'height': height, 'nodata':self.header['NODATA_value']})
        if output_file is None:
            filename = 'temp.tif'
        else:
            if output_file.endswith('.tif'):
                filename = output_file
            else:
                filename = output_file+'.tif'
        dst_rio = rasterio.open(filename, 'w+', **kwargs)
        reproject(source=rasterio.band(src_rio, 1),
                  destination=rasterio.band(dst_rio, 1),
                  src_transform=src_rio.transform,
                  src_crs=src_rio.crs,
                  dst_transform=transform,
                  dst_crs=dst_crs,
                  resampling=Resampling.nearest)
        if output_file is not None:
            dst_rio.close()
        return dst_rio
#%%=============================Visualization==================================
    def mapshow(self, **kwargs):
        """Display raster data without projection

        Args:
            figname: str, the file name to export map
            figsize: tuple, the size of map
            dpi: scalar, The resolution in dots per inch
            cax_str: str, the title of the colorbar
            relocate: True|False, relocate the origin of the grid to (0, 0)
            scale_ratio: 1|1000, axis unit 1 m or 1000 meter
            vmin: define the data range that the colormap covers
            vmax: define the data range that the colormap covers
            ytick_labelrotation: degree to rotate tick labels on y axis
            
        Example:
             mapshow(ax=ax, figname='my_fig', figsize=(6, 8), dpi=300,
                     title='My map', cax=True, cax_str='Meter', 
                     relocate=False, scale_ratio=1000, 
                     ytick_labelrotation=90)
        """
        fig, ax = gs.mapshow(raster_obj=self, **kwargs)
        return fig, ax
    
    def rankshow(self, **kwargs):
        """ Display array values in ranks
        
        Args:
            breaks: list of values to define rank. Array values lower than the
                first break value are set as nodata.
            color: color series of the ranks
            ylabelrotation: scalar giving degree to rotate yticklabel
            legend_kw: dict, keyword arguments to set legend. A colobar 
                     rather than a legend be displayed  if legend_kw is None.
            **kwargs: keywords argument of function imshow
        
        Example:
            rankshow(figname=None, figsize=None, 
                     dpi=200, ax=None, color='Blues', 
                     breaks=[0.2, 0.3, 0.5, 1, 2],
                     legend_kw={'loc':'upper left', 'facecolor':None, 
                                'fontsize':'small', 'title':'depth(m)', 
                                'labelspacing':0.1}, 
                     ytick_labelrotation=None, relocate=False, scale_ratio=1,
                     alpha=1)
        
        """
        fig, ax = gs.rankshow(self, **kwargs)
        return fig, ax
    
    def hillshade(self, **kwargs):
        """ Draw a hillshade map
        """
        fig, ax = gs.hillshade(self, **kwargs)
        return fig, ax

    def vectorshow(self, obj_y, **kwargs):
        """
        plot velocity map of U and V, whose values stored in two raster
        objects seperately
        """
        fig, ax = gs.vectorshow(self, obj_y, **kwargs)
        return fig, ax
    
#%%
def merge(obj_origin, obj_target, resample_method='bilinear'):
    """Merge the obj_origin to obj_target

    assign grid values in the origin Raster to the cooresponding grid cells in
    the target object. If cellsize are not equal, the origin Raster will be
    firstly resampled to the target object.
    
    Args:
        obj_origin: (Raster) original raster
        obj_target: (Raster) target raster
    """
    if obj_origin.header['cellsize'] != obj_target.header['cellsize']:
        obj_origin = obj_origin.resample(obj_target.header['cellsize'], 
                                   method=resample_method)
    grid_x, grid_y = obj_origin.to_points()
    rows, cols = sp.map2sub(grid_x, grid_y, obj_target.header)
    ind_r = np.logical_and(rows >= 0, rows <= obj_target.header['nrows']-1)
    ind_c = np.logical_and(cols >= 0, cols <= obj_target.header['ncols']-1)
    ind = np.logical_and(ind_r, ind_c)
    ind = np.logical_and(ind, ~np.isnan(obj_origin.array))
    obj_output = copy.deepcopy(obj_target)
    obj_output.array[rows[ind], cols[ind]] = obj_origin.array[ind]
    return obj_output

def main():
    """Main function
    """
    print('Class to deal with raster data')

if __name__=='__main__':
    main()
    
    

