import unittest
import numpy as np
import os
import shutil

from synxflow.IO.Landcover import Landcover
from synxflow.IO.Raster import Raster
from synxflow.IO.spatial_analysis import arcgridwrite

# Define dummy data at module level or in setUpClass for clarity
DUMMY_ROWS, DUMMY_COLS = 4, 5
DUMMY_CELLSIZE = 10.0
DUMMY_XLL, DUMMY_YLL = 0.0, 0.0
DUMMY_NODATA = -9999.0

# DEM Array (all valid for simplicity)
DUMMY_DEM_ARRAY = np.ones((DUMMY_ROWS, DUMMY_COLS)) * 10.0
DUMMY_DEM_HEADER = {
    'ncols': DUMMY_COLS, 'nrows': DUMMY_ROWS,
    'xllcorner': DUMMY_XLL, 'yllcorner': DUMMY_YLL,
    'cellsize': DUMMY_CELLSIZE, 'NODATA_value': DUMMY_NODATA
}
# Create a dummy DEM Raster object needed for Landcover init context
DUMMY_DEM_RASTER = Raster(array=DUMMY_DEM_ARRAY, header=DUMMY_DEM_HEADER)

# Landcover Array (indices 0, 1, and a NoData cell)
# Use integers for landcover types, float for array to hold NODATA
DUMMY_LANDCOVER_ARRAY = np.array([
    [0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0],
    [0, DUMMY_NODATA, 1, 1, 0], # Add a NoData value matching DEM's nodata
    [0, 0, 1, 1, 1]
]).astype(float)
# Create a dummy Landcover Raster object
DUMMY_LANDCOVER_RASTER = Raster(array=DUMMY_LANDCOVER_ARRAY, header=DUMMY_DEM_HEADER)


class TestLandcover(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = f"temp_test_landcover_{self.id().split('.')[-1]}"
        os.makedirs(self.test_dir, exist_ok=True)

        # Write dummy landcover file
        self.dummy_landcover_path = os.path.join(self.test_dir, "dummy_landcover.asc")
        # Use the class-level array and header defined above
        arcgridwrite(self.dummy_landcover_path, DUMMY_LANDCOVER_ARRAY, DUMMY_DEM_HEADER)

        # Store copies of dummy data for modification tests if needed
        self.landcover_array = DUMMY_LANDCOVER_ARRAY.copy()
        self.dem_raster = Raster(array=DUMMY_DEM_ARRAY.copy(), header=DUMMY_DEM_HEADER.copy())
        self.landcover_raster = Raster(array=DUMMY_LANDCOVER_ARRAY.copy(), header=DUMMY_DEM_HEADER.copy())


    def tearDown(self):
        """Tear down test fixtures after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    # --- Initialization Tests ---
    def test_init_with_raster_and_dem(self):
        """Test Landcover initialization with Raster objects for landcover and DEM."""
        landcover = Landcover(ras_data=self.landcover_raster, dem_ras=self.dem_raster)

        self.assertIsInstance(landcover, Landcover)
        self.assertEqual(landcover.mask_header, self.dem_raster.header, "Header should match DEM header")
        # Check mask_dict structure (basic check)
        self.assertIsInstance(landcover.mask_dict, dict)
        self.assertTrue('value' in landcover.mask_dict)
        self.assertTrue('index' in landcover.mask_dict)
        # Check unique values stored (should be 0, 1, NODATA or just 0, 1 depending on _mask2dict impl.)
        # Assuming NODATA is filtered out by _mask2dict based on Raster structure
        np.testing.assert_array_equal(np.sort(landcover.mask_dict['value']), np.array([0, 1]))
        # Check subs_in (should be populated because dem_ras was provided)
        self.assertIsNotNone(getattr(landcover, 'subs_in', None))
        self.assertEqual(len(landcover.subs_in), 2) # Should be tuple of (rows, cols)

    def test_init_with_filepath_and_dem(self):
        """Test Landcover initialization with a file path and DEM Raster."""
        landcover = Landcover(ras_data=self.dummy_landcover_path, dem_ras=self.dem_raster)

        self.assertIsInstance(landcover, Landcover)
        self.assertEqual(landcover.mask_header, self.dem_raster.header)
        np.testing.assert_array_equal(np.sort(landcover.mask_dict['value']), np.array([0, 1]))
        self.assertIsNotNone(getattr(landcover, 'subs_in', None))

    def test_init_without_dem(self):
        """Test Landcover initialization without providing dem_ras."""
        landcover = Landcover(ras_data=self.landcover_raster) # No dem_ras

        self.assertIsInstance(landcover, Landcover)
        # Header should match the landcover raster itself
        self.assertEqual(landcover.mask_header, self.landcover_raster.header)
        # subs_in should not be set
        self.assertFalse(hasattr(landcover, 'subs_in') and landcover.subs_in is not None, "subs_in should not be set without dem_ras")
        np.testing.assert_array_equal(np.sort(landcover.mask_dict['value']), np.array([0, 1]))

    # --- Getter Method Tests ---
    def test_get_mask_array(self):
        """Test reconstructing the mask array."""
        landcover = Landcover(ras_data=self.landcover_raster, dem_ras=self.dem_raster)
        mask_array = landcover.get_mask_array()

        self.assertEqual(mask_array.shape, self.landcover_array.shape)
        # Compare, handling the NODATA value which becomes NaN in Raster arrays
        expected_array = self.landcover_array.copy()
        expected_array[expected_array == DUMMY_NODATA] = np.nan
        expected_array_filled = expected_array.copy()
        expected_array_filled[np.isnan(expected_array_filled)] = 0 # Assume fills with 0
        np.testing.assert_allclose(mask_array, expected_array_filled.astype(mask_array.dtype), equal_nan=True)


    # --- Parameter Generation Tests ---
    def test_to_grid_parameter_scalar_value(self):
        """Test generating grid parameters with a single value mapping."""
        landcover = Landcover(ras_data=self.landcover_raster, dem_ras=self.dem_raster)
        param_value = 0.5  # Assign this value
        land_ids = [1]     # To cells where landcover is 1
        default_value = 0.1 # For all other cells

        param_grid = landcover.to_grid_parameter(param_value=param_value,
                                                 land_value=land_ids,
                                                 default_value=default_value)

        self.assertEqual(param_grid.shape, self.landcover_array.shape)
        # Create expected grid
        expected_grid = np.full_like(self.landcover_array, default_value, dtype=float)
        # Apply param_value where landcover matches land_ids
        original_mask = self.landcover_array
        expected_grid[original_mask == 1] = param_value
        expected_grid[original_mask == DUMMY_NODATA] = default_value

        np.testing.assert_allclose(param_grid, expected_grid)

    def test_to_grid_parameter_list_values(self):
        """Test generating grid parameters with multiple value mappings."""
        landcover = Landcover(ras_data=self.landcover_raster, dem_ras=self.dem_raster)
        param_values = [0.5, 0.8]  # Assign 0.5 to land type 1, 0.8 to land type 0
        land_ids_list = [[1], [0]] # Corresponding landcover IDs for each param_value
        default_value = 0.1        # For any other land types (or NoData)

        param_grid = landcover.to_grid_parameter(param_value=param_values,
                                                 land_value=land_ids_list,
                                                 default_value=default_value)

        self.assertEqual(param_grid.shape, self.landcover_array.shape)
        # Create expected grid
        expected_grid = np.full_like(self.landcover_array, default_value, dtype=float)
        original_mask = self.landcover_array
        # Apply mappings
        expected_grid[np.isin(original_mask, land_ids_list[0])] = param_values[0] # Where mask is 1, value is 0.5
        expected_grid[np.isin(original_mask, land_ids_list[1])] = param_values[1] # Where mask is 0, value is 0.8
        nodata_indices = (original_mask == DUMMY_NODATA)
        # Assign the value corresponding to landcover type 0 (which is 0.8 in this test case)
        value_for_type_0 = param_values[land_ids_list.index([0])] # Get the value mapped to [0]
        expected_grid[nodata_indices] = value_for_type_0 # Set NoData locations to 0.8

        np.testing.assert_allclose(param_grid, expected_grid)

    def test_to_grid_parameter_multiple_ids_per_value(self):
        """Test assigning one parameter value to multiple landcover IDs."""
        landcover = Landcover(ras_data=self.landcover_raster, dem_ras=self.dem_raster)
        param_value = 0.9  # Assign this value
        land_ids = [0, 1]  # To cells where landcover is 0 OR 1
        default_value = 0.1

        param_grid = landcover.to_grid_parameter(param_value=param_value,
                                                 land_value=land_ids,
                                                 default_value=default_value)

        self.assertEqual(param_grid.shape, self.landcover_array.shape)
        # Create expected grid
        expected_grid = np.full_like(self.landcover_array, default_value, dtype=float)
        original_mask = self.landcover_array
        # Apply param_value where landcover matches any in land_ids
        expected_grid[np.isin(original_mask, land_ids)] = param_value
        # Handle NoData cells - assume default
        nodata_indices = (original_mask == DUMMY_NODATA)
        # Assign the value corresponding to landcover type 0 (which is 0.8 in this test case)
        value_for_type_0 = param_value # Get the value mapped to [0]
        expected_grid[nodata_indices] = value_for_type_0 # Set NoData locations to 0.8

        np.testing.assert_allclose(param_grid, expected_grid)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)