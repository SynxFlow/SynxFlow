#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Created on Tue Mar 31 16:20:13 2020
# Author: Xiaodong Ming

"""
Raster
=======

To do:
    Read, write and analyze gridded Raster data

---------------

"""
import copy
import math
import fiona
import numpy as np
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
        projection: (string) the Well-Known_Text (wkt) projection information  
        
    """
#%%======================== initialization function ===========================   
    def __init__(self, source_file=None, array=None, header=None, 
                 xllcorner=0, yllcorner=0, cellsize=100, NODATA_value=-9999,
                 epsg=None, projection=None, num_header_rows=6):
        """Initialise the object

        Args:
            source_file: name of a asc/tif file if a file read is needed
            array: values in each raster cell [a numpy array]
            header: georeference of the raster [a dictionary containing 6 keys]
                nrows, nclos [int]
                cellsize, xllcorner, yllcorner
                NODATA_value
            epsg: epsg code [int]
            projection: WktProjection [string]

        """
        if epsg is not None:
            projection = self.__set_wkt_projection(epsg)
        if type(source_file) is str:
            if source_file.endswith('.tif'):
                array, header, projection = sp.tif_read(source_file) 
                # only read the first band
            else:
                array, header, projection = sp.arcgridread(source_file,
                                                    num_header_rows)
            self.source_file = source_file
        elif type(source_file) is bytes:  # try a binary file-like object
            array, header = sp.byte_file_read(source_file)
        if header is None:
            header = {'ncols':array.shape[1], 'nrows':array.shape[0],
                      'xllcorner':xllcorner, 'yllcorner':yllcorner,
                      'cellsize':cellsize, 'NODATA_value':NODATA_value}
        self.header = header
        extent = sp.header2extent(header)
        self.source_file = source_file
        self.projection = projection
        self.array = array
        self.shape = array.shape
        if array.shape != (header['nrows'], header['ncols']):
            raise ValueError(('shape of array is not consistent with '
                              'nrows and ncols in header'))
        self.cellsize = header['cellsize']
        self.extent = extent
        self.extent_dict = {'left':extent[0], 'right':extent[1],
                            'bottom':extent[2], 'top':extent[3]}
            
#%%============================= Spatial analyst ==============================   
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
        array_new = self.array[min(rows):max(rows), min(cols):max(cols)]
        header_new['nrows'] = array_new.shape[0]
        header_new['ncols'] = array_new.shape[1]
        header_new['xllcorner'] = xllcorner
        header_new['yllcorner'] = yllcorner
        new_obj = Raster(array=array_new, header=header_new,
                         projection=self.projection)
        return new_obj
    
    def clip(self, clip_mask=None):
        """clip raster according to a mask

        Args:
            mask: 1. string name of a shapefile or 2. 2-col numpy array giving
                X and Y coords in each column to shape the mask polygon
        
        Return:
            Raster: a new raster object
        """
        from rasterio import mask
        ds_rio = self.to_rasterio()
        if type(clip_mask) is str:
            with fiona.open(clip_mask, 'r') as shapefile:
                shapes = [feature['geometry'] for feature in shapefile]
        elif type(clip_mask) is np.ndarray:
            shape = {'type':'Polygon', 'coordinates':[clip_mask]}
            shapes = [shape]
        else:
            raise IOError('mask must be either a string or a numpy array')
        
        out_image, out_transform = mask.mask(ds_rio, shapes, crop=True) # 
        array = out_image[0]
        cellsize = out_transform[0]
        xllcorner = out_transform[2]
        yllcorner = out_transform[5]- cellsize*array.shape[0]
        obj_clipped = Raster(array=array, xllcorner=xllcorner,
                             yllcorner=yllcorner, NODATA_value=ds_rio.nodata,
                             cellsize=cellsize)
        return obj_clipped
    
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
        ds_rio = self.to_rasterio()
        if type(shp_filename) is str:
            with fiona.open(shp_filename, 'r') as shapefile:
                shapes = [feature['geometry'] for feature in shapefile]
        else:
            shapes = shp_filename
        out_image, _ = mask.mask(ds_rio, shapes) #, crop=True
        rasterized_array = out_image[0]
        rasterized_array[np.isnan(rasterized_array)] = ds_rio.nodata
        index_array = np.full(rasterized_array.shape, True)
        index_array[rasterized_array == ds_rio.nodata] = False
#        ds_rio.close()
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
        ds_rio = self.to_rasterio()
        new_shape = (1, int(ds_rio.height * upscale_factor),
                        int(ds_rio.width * upscale_factor))
        resampling_method = Resampling(ind)
        data = ds_rio.read(out_shape=new_shape, resampling=resampling_method)
        data = data[0]
        # scale image transform
        transform = ds_rio.transform * ds_rio.transform.scale(
                    (ds_rio.width / data.shape[-1]),
                    (ds_rio.height / data.shape[-2]))
        new_header = copy.deepcopy(self.header)
        new_header['cellsize'] = new_cellsize
        new_header['nrows'] = data.shape[0]
        new_header['ncols'] = data.shape[1]
        new_header['xllcorner'] = transform[2]
        new_header['yllcorner'] = transform[5]-data.shape[0]*new_cellsize
        obj_new = Raster(array=data, header=new_header)
#        ds_rio.close()
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
        new_obj = copy.deepcopy(self)
        new_obj.array = array_interp
        new_obj.source_file = 'mask_'+new_obj.source_file
        return new_obj
    
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
        new_obj = Raster(array=array, header=header)
        return new_obj
    
    def assign_to(self, new_header):
        """ Assign_to the object to a new grid defined by new_header 

        If cellsize are not equal, the origin Raster will be firstly 
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
        obj_output = Raster(array=new_array, header=new_header)
        return obj_output

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
    
    def write_asc(self, output_file, EPSG=None, compression=False):
        """ write raster as asc format file

        Alias: Write_asc

        Args:
            output_file: output file name
            EPSG: epsg code, if it is given, a .prj file will be written
            compression: logic, whether compress write the asc file as gz
        """
        sp.arcgridwrite(output_file, self.array, self.header, compression)
        if EPSG is not None:
            self.__set_wkt_projection(EPSG)
        # if projection is defined, write .prj file for asc file
        if output_file.endswith('.asc'):
            if self.projection is not None:
                prj_file=output_file[0:-4]+'.prj'
                wkt = self.projection
                with open(prj_file, "w") as prj:        
                    prj.write(wkt)
        return None
    
    def to_osgeo_raster(self, filename=None, fileformat = 'GTiff',
                        srcEPSG=27700, destEPSG=None):        
        """convert this object to an osgeo raster object, write a tif file if 
        necessary

        Args:
            filename: the output file name
            fileformat: GTiff or AAIGrid
            destEPSG: the EPSG projection code default: 27700 British National
                Grid 'EPSG:4326'

        Returns:
            osgeo raster: The converted raster dataset, or a tif filename if 
                it is written

        Alias: To_osgeo_raster
        """
        from osgeo import gdal, osr
        if filename is None:
            dst_filename = ''
            driver_name = 'MEM'
        else:
            dst_filename = filename
            driver_name = fileformat
        if not dst_filename.endswith('.tif'):
            dst_filename = dst_filename+'.tif'
    
        # You need to get those values like you did.
        PIXEL_SIZE = self.header['cellsize']  # size of the pixel...        
        x_min = self.extent[0] # left  
        y_max = self.extent[3] # top
        src_crs = osr.SpatialReference()
        src_crs.ImportFromEPSG(srcEPSG)
        # create dataset with driver
        driver = gdal.GetDriverByName(driver_name)
        ncols = int(self.header['ncols'])
        nrows = int(self.header['nrows'])
        dataset = driver.Create(dst_filename, 
            xsize=ncols, 
            ysize=nrows, 
            bands=1, 
            eType=gdal.GDT_Float32)
    
        dataset.SetGeoTransform((
            x_min,       # 0
            PIXEL_SIZE,  # 1
            0,           # 2
            y_max,       # 3
            0,           # 4
            -PIXEL_SIZE))  
    
        dataset.SetProjection(src_crs.ExportToWkt())
        array = self.array
        dataset.GetRasterBand(1).WriteArray(array)
        dataset.GetRasterBand(1).SetNoDataValue(self.header['NODATA_value'])
        if filename is not None:
            if destEPSG is not None:
#                dest_crs = osr.SpatialReference()
#                dest_crs.ImportFromEPSG(destEPSG)
                gdal.Warp(dst_filename, dataset, dstSRS=destEPSG)
            dataset.FlushCache()  # Write to disk.
            dataset = None
            return dst_filename
        else:
            return dataset
    
    def to_rasterio(self, output_file=None, src_epsg=27700):
        """ Convert to a rasterio dataset
        
        Args:
            output_file: a string to give output file name
            src_epsg: int scalar to give EPSG code of the coordinate reference
                system of the original dataset, default is 27700 for BNG
        Return:
            ds_rio: a rasterio dataset
        """
        import rasterio
        from rasterio.transform import Affine
        cellsize = self.cellsize
        x00 = self.extent_dict['left'] # upper-left corner of the first pixel
        y00 = self.extent_dict['top']
        transform = Affine.translation(x00, y00)*Affine.scale(
                                                       cellsize, -cellsize)
        if output_file is None:
            filename = '/tmp/new.tif'
        else:
            if output_file.endswith('.tif'):
                filename = output_file
            else:
                filename = output_file+'.tif'
        src_crs = rasterio.crs.CRS.from_epsg(src_epsg)
        ds_rio = rasterio.open(filename, 'w+', driver='GTiff',
                               height=self.shape[0], width=self.shape[1],
                               count=1,
                               dtype=self.array.dtype,
                               crs=src_crs,
                               transform=transform,
                               nodata=self.header['NODATA_value'])
        ds_rio.write(self.array, 1)
        if output_file is not None:
            ds_rio.close()
        return ds_rio
    
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
        src_rio = self.to_rasterio()
        dst_crs = rasterio.crs.CRS.from_epsg(dst_epsg)
        transform, width, height = cal_tsf(src_rio.crs, dst_crs, src_rio.width, 
                                           src_rio.height, *src_rio.bounds)
        kwargs = src_rio.meta.copy()
        kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width,
                       'height': height, 'nodata':self.header['NODATA_value']})
        if output_file is None:
            filename = '/tmp/new.tif'
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
            figname: the file name to export map
            figsize: the size of map
            dpi: The resolution in dots per inch
            vmin and vmax define the data range that the colormap covers
            cax_str: string as the title of the colorbar
            relocate: relocate the origin of the grid coordinates to (0, 0)
            scale_ratio: 1|1000, axis unit 1 m or 1000 meter
        """
        fig, ax = gs.mapshow(raster_obj=self, **kwargs)
        return fig, ax
    
    def rankshow(self, **kwargs):
        """ Display water depth map in a range defined by (d_min, d_max)
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
#%%=========================== private functions ==============================
    def __osgeo2raster(self, obj_ds):
        """
        convert an osgeo dataset to a raster object
        """
        array = obj_ds.ReadAsArray()
        geo_trans_v = obj_ds.GetGeoTransform()
        projection = obj_ds.GetProjection()
        left = geo_trans_v[0]
        top = geo_trans_v[3]
        cellsize = geo_trans_v[1]
        nrows = obj_ds.RasterYSize
        ncols = obj_ds.RasterXSize
        xllcorner = left
        yllcorner = top - cellsize*nrows
        NODATA_value = obj_ds.GetRasterBand(1).GetNoDataValue()
        if NODATA_value is None:
            NODATA_value = -9999
        header = {'ncols':ncols, 'nrows':nrows,
                  'xllcorner':xllcorner, 'yllcorner':yllcorner,                  
                  'cellsize':cellsize, 'NODATA_value':NODATA_value}
        obj_new = Raster(array=array, header=header, projection=projection)
        return obj_new

    def __set_wkt_projection(self, epsg_code):
        """
        get coordinate reference system (crs) as Well Known Text (WKT) 
            from https://epsg.io

        epsg_code: the epsg code of a crs, e.g. BNG:27700, WGS84:4326

        return wkt text
        """
        import requests
        # access projection information
        wkt = requests.get('https://epsg.io/{0}.prettywkt/'.format(epsg_code))
        # remove spaces between charachters
        remove_spaces = wkt.text.replace(" ", "")
        # place all the text on one line
        output = remove_spaces.replace("\n", "")
        self.projection = output
        return output
    
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
#    else:
#        obj_origin = self
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
    
    

