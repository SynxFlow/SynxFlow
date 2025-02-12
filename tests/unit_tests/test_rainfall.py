"""
Test module for Rainfall class in synxflow package.
Tests both successful data loading scenarios and error handling cases using sample data.

The test suite validates:
1. Sample data loading and processing
2. Error handling for invalid file paths
3. Error handling for mismatched data shapes
4. Error handling for invalid data types
"""

import pytest
import os
import numpy as np
from synxflow.IO.Rainfall import Rainfall
from synxflow.IO.Raster import Raster
from synxflow.IO.demo_functions import get_sample_data

def test_load_sample_rainfall_data():
    """Test successful loading and processing of sample rainfall data from package.
    
    Tests:
        - Loading of DEM, rain mask, and rain source files
        - Creation of Rainfall object with sample data
        - Verification of essential attributes in the Rainfall object
    
    Expected Results:
        - All required attributes should exist
        - No exceptions should be raised
    """
    # Get paths to sample data included in the package
    dem_file, demo_data, data_path = get_sample_data(case_type='flood')
    
    # Load DEM and create Raster object from sample data
    dem_ras = Raster(os.path.join(data_path, 'DEM.gz'))
    
    # Define paths to rainfall mask and source files
    rain_mask = os.path.join(data_path, 'rain_mask.gz')
    rain_source = os.path.join(data_path, 'rain_source.csv')
    
    # Create and initialize Rainfall object with sample data
    rainfall = Rainfall(rain_mask, rain_source, dem_ras=dem_ras)
    
    # Verify all required attributes exist and are properly initialized
    assert rainfall.mask_header is not None, "Mask header not initialized"
    assert hasattr(rainfall, 'time_s'), "Time series attribute missing"
    assert hasattr(rainfall, 'rain_rate'), "Rain rate attribute missing"
    assert hasattr(rainfall, 'subs_in'), "Subscripts attribute missing"

def test_invalid_sample_data_path():
    """Test error handling when trying to load non-existent data files.
    
    Tests:
        - Attempt to create Rainfall object with non-existent mask file
        - Verify appropriate exception is raised
    
    Expected Results:
        - Should raise FileNotFoundError or IOError
    """
    # Get sample data paths for valid DEM
    dem_file, demo_data, data_path = get_sample_data(case_type='flood')
    dem_ras = Raster(os.path.join(data_path, 'DEM.gz'))
    
    # Attempt to create Rainfall object with non-existent mask file
    with pytest.raises((FileNotFoundError, IOError)):
        rainfall = Rainfall('nonexistent_mask.gz', 
                          os.path.join(data_path, 'rain_source.csv'),
                          dem_ras=dem_ras)

def test_mismatched_rainfall_mask_shape():
    """Test error handling when rainfall mask shape doesn't match DEM dimensions.
    
    Tests:
        - Attempt to use rainfall mask with incorrect dimensions
        - Verify appropriate error message is raised
    
    Expected Results:
        - Should raise ValueError with specific error message about shape mismatch
    """
    # Load sample DEM data
    dem_file, demo_data, data_path = get_sample_data(case_type='flood')
    dem_ras = Raster(os.path.join(data_path, 'DEM.gz'))
    
    # Create intentionally wrong-sized mask array
    wrong_shape_mask = np.ones((10, 10))  # Different dimensions than DEM
    rain_source = os.path.join(data_path, 'rain_source.csv')
    
    # Attempt to create Rainfall object with mismatched dimensions
    with pytest.raises(ValueError, 
                      match="The shape of rainfall_mask array is not consistent with DEM"):
        rainfall = Rainfall(wrong_shape_mask, rain_source, dem_ras=dem_ras)

def test_invalid_rain_source_type():
    """Test error handling when rain source has invalid data type.
    
    Tests:
        - Attempt to use invalid data type (dictionary) for rain source
        - Verify appropriate error message is raised
    
    Expected Results:
        - Should raise IOError with specific error message about invalid source type
    """
    # Load sample data
    dem_file, demo_data, data_path = get_sample_data(case_type='flood')
    dem_ras = Raster(os.path.join(data_path, 'DEM.gz'))
    rain_mask = os.path.join(data_path, 'rain_mask.gz')
    
    # Create invalid rain source using dictionary instead of array/filename
    invalid_source = {'time': [0, 3600], 'rate': [0.001, 0.002]}
    
    # Attempt to create Rainfall object with invalid source type
    with pytest.raises(IOError, 
                      match="rain_source must be either a filename or a numpy array"):
        rainfall = Rainfall(rain_mask, invalid_source, dem_ras=dem_ras)

if __name__ == '__main__':
    # Execute all test cases when script is run directly
    test_load_sample_rainfall_data()
    test_invalid_sample_data_path()
    test_mismatched_rainfall_mask_shape()
    test_invalid_rain_source_type()
    print("All tests passed!")