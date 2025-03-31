"""
Test module for Raster class in synxflow package.

This test suite validates the core functionality of the Raster class including:
1. Object initialization with arrays and headers
2. Extent calculations
3. Error handling for invalid inputs
4. CRS (Coordinate Reference System) handling
5. Mask operations

Each test function focuses on a specific aspect of the Raster class functionality
and includes appropriate assertions and error checking.
"""

import pytest
import numpy as np
from synxflow.IO.Raster import Raster

def test_raster_init_with_array():
    """Test basic initialization of a Raster object with array and header.
    
    Tests:
        - Creation of Raster object with valid input data
        - Array shape validation
        - Array content validation
        - Header information validation
    
    Expected Results:
        - Object should be created successfully
        - Array dimensions should match input
        - Array values should match input
        - Header should match input dictionary
    """
    # Create test data with known values
    test_array = np.array([[1, 2, 3], [4, 5, 6]])  # Simple 2x3 array
    test_header = {
        'ncols': 3,          # Number of columns in the grid
        'nrows': 2,          # Number of rows in the grid
        'xllcorner': 0,      # Lower left X coordinate
        'yllcorner': 0,      # Lower left Y coordinate
        'cellsize': 1,       # Size of each grid cell
        'NODATA_value': -9999  # Value representing no data
    }
    
    # Initialize Raster object with test data
    raster = Raster(array=test_array, header=test_header)
    
    # Verify the object's properties match input data
    assert raster.array.shape == (2, 3), "Array shape does not match input dimensions"
    assert np.array_equal(raster.array, test_array), "Array values do not match input"
    assert raster.header == test_header, "Header does not match input dictionary"

def test_raster_extent_calculation():
    """Test the calculation of raster spatial extent.
    
    Tests:
        - Extent calculation based on header information
        - Verification of all extent boundaries
    
    Expected Results:
        - Extent values should match calculated boundaries
        - All four boundaries (xmin, xmax, ymin, ymax) should be correct
    """
    # Create test data with non-zero corners and cell size
    test_array = np.array([[1, 2, 3], [4, 5, 6]])
    test_header = {
        'ncols': 3,
        'nrows': 2,
        'xllcorner': 10,     # Start at X=10
        'yllcorner': 20,     # Start at Y=20
        'cellsize': 5,       # Each cell is 5x5 units
        'NODATA_value': -9999
    }
    
    # Create Raster object and get its extent
    raster = Raster(array=test_array, header=test_header)
    extent = raster.extent
    
    # Verify each boundary of the extent
    assert extent[0] == 10, "Incorrect minimum X coordinate"  # xmin
    assert extent[1] == 25, "Incorrect maximum X coordinate"  # xmax = xmin + (ncols * cellsize)
    assert extent[2] == 20, "Incorrect minimum Y coordinate"  # ymin
    assert extent[3] == 30, "Incorrect maximum Y coordinate"  # ymax = ymin + (nrows * cellsize)

def test_raster_mismatched_dimensions():
    """Test error handling for mismatched array and header dimensions.
    
    Tests:
        - Error raising when array shape doesn't match header specifications
    
    Expected Results:
        - Should raise ValueError with specific error message
    """
    # Create test data with intentionally mismatched dimensions
    test_array = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3 array
    test_header = {
        'ncols': 4,  # Incorrect number of columns (should be 3)
        'nrows': 2,
        'xllcorner': 0,
        'yllcorner': 0,
        'cellsize': 1,
        'NODATA_value': -9999
    }
    
    # Verify that initialization raises appropriate error
    with pytest.raises(ValueError, 
                      match="shape of array is not consistent with nrows and ncols in header"):
        raster = Raster(array=test_array, header=test_header)

def test_invalid_mask_type():
    """Test error handling for invalid mask type in clipping operation.
    
    Tests:
        - Error raising when invalid mask type is provided for clipping
    
    Expected Results:
        - Should raise IOError with specific error message
    """
    # Create valid Raster object first
    test_array = np.array([[1, 2, 3], [4, 5, 6]])
    test_header = {
        'ncols': 3,
        'nrows': 2,
        'xllcorner': 10,
        'yllcorner': 20,
        'cellsize': 5,
        'NODATA_value': -9999
    }
    raster = Raster(array=test_array, header=test_header)
    
    # Test with invalid mask type (dictionary instead of array/string)
    with pytest.raises(IOError, match='mask must be either a string or a numpy array'):
        raster.clip(clip_mask={'invalid': 'mask'})

def test_array_header_shape_mismatch():
    """Test error handling for array and header shape mismatch.
    
    Tests:
        - Error raising when array dimensions don't match header specifications
    
    Expected Results:
        - Should raise ValueError with specific error message
    """
    # Create test data with mismatched dimensions
    test_array = np.array([[1, 2], [3, 4]])  # 2x2 array
    test_header = {
        'ncols': 3,  # Header specifies 3 columns but array has 2
        'nrows': 2,
        'xllcorner': 0,
        'yllcorner': 0,
        'cellsize': 1,
        'NODATA_value': -9999
    }

    # Verify initialization raises appropriate error
    with pytest.raises(ValueError, 
                      match='shape of array is not consistent with nrows and ncols in header'):
        Raster(array=test_array, header=test_header)

def test_invalid_crs_type():
    """Test error handling for invalid CRS (Coordinate Reference System) type.
    
    Tests:
        - Error raising when invalid CRS type is provided
    
    Expected Results:
        - Should raise IOError with specific error message
    """
    # Create valid Raster object
    test_array = np.array([[1, 2], [3, 4]])
    test_header = {
        'ncols': 2,
        'nrows': 2,
        'xllcorner': 0,
        'yllcorner': 0,
        'cellsize': 1,
        'NODATA_value': -9999
    }
    raster = Raster(array=test_array, header=test_header)
    
    # Attempt to set CRS with invalid type
    invalid_crs = [1, 2, 3]  # List instead of valid CRS type
    
    # Verify setting invalid CRS raises appropriate error
    with pytest.raises(IOError, match='crs must be int|string|rasterio.crs object'):
        raster.set_crs(invalid_crs)

if __name__ == '__main__':
    # Run all tests when script is executed directly
    test_raster_init_with_array()
    test_raster_extent_calculation()
    test_raster_mismatched_dimensions()
    test_invalid_mask_type()
    test_array_header_shape_mismatch()
    test_invalid_crs_type()
    print("All raster tests completed successfully!")