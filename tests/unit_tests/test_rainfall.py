import unittest
import numpy as np
import os
import shutil
import sys # Keep for skipIf

from synxflow.IO.Rainfall import Rainfall
from synxflow.IO.Raster import Raster
from synxflow.IO.demo_functions import get_sample_data

class TestRainfallRefocused(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load sample data paths and base DEM once."""
        print("Setting up class: Loading sample data paths...")
        try:
            # Get paths to sample data included in the package
            cls.dem_file, cls.demo_data_dict, cls.data_path = get_sample_data(case_type='flood')
            cls.sample_mask_path = os.path.join(cls.data_path, 'rain_mask.gz')
            cls.sample_source_path = os.path.join(cls.data_path, 'rain_source.csv')
            # Load the base DEM raster for context
            cls.base_dem_raster = Raster(cls.dem_file)
            cls.sample_data_available = True
        except Exception as e:
            print(f"Warning: Could not load sample data via get_sample_data: {e}")
            cls.sample_data_available = False
            # Define dummy raster if needed for tests that don't rely on loading sample
            cls.base_dem_raster = Raster(array=np.ones((4,5)), header={'ncols': 5, 'nrows': 4,'xllcorner': 0, 'yllcorner': 0,'cellsize': 10, 'NODATA_value': -9999.0})


    def setUp(self):
        """Set up test fixtures before each test method (minimal setup needed now)."""
        # Create a temporary directory for potential temporary file needs (if any test needed it)
        self.test_dir = f"temp_test_rainfall_refocused_{self.id().split('.')[-1]}"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest.skipUnless(lambda: hasattr(TestRainfallRefocused, 'sample_data_available') and TestRainfallRefocused.sample_data_available,
                         "Sample data not loaded, skipping sample load test")
    def test_load_sample_rainfall_data(self):
        """Test successful loading and processing of sample rainfall data from package."""
        # Uses class variables set up in setUpClass
        rainfall = Rainfall(rain_mask=self.sample_mask_path,
                            rain_source=self.sample_source_path,
                            source_sep=',', # Specify separator for csv
                            dem_ras=self.base_dem_raster)

        # Verify essential attributes exist and seem reasonable
        self.assertIsNotNone(rainfall.mask_header, "Mask header not initialized")
        self.assertTrue(hasattr(rainfall, 'time_s'), "Time series attribute missing")
        self.assertTrue(hasattr(rainfall, 'rain_rate'), "Rain rate attribute missing")
        self.assertTrue(hasattr(rainfall, 'subs_in'), "Subscripts attribute missing (from dem_ras)")
        self.assertTrue(hasattr(rainfall, 'attrs'), "Attributes dict missing")
        self.assertIsInstance(rainfall.attrs, dict)
        self.assertTrue(len(rainfall.time_s) > 0, "Time series is empty")
        self.assertTrue(rainfall.rain_rate.shape[0] == len(rainfall.time_s), "Rain rate rows != time series length")
        self.assertTrue(rainfall.rain_rate.shape[1] > 0, "Rain rate has no sources")


    def test_init_invalid_mask_path(self):
        """Test error handling for non-existent mask file path."""
        non_existent_mask = os.path.join(self.test_dir, "non_existent_mask.gz")
        # Use a valid source path if available, otherwise skip or use dummy valid path concept
        valid_source = getattr(self, 'sample_source_path', 'dummy_valid_source.csv')
        if not self.sample_data_available and valid_source == 'dummy_valid_source.csv':
             # Create dummy valid source file if sample wasn't loaded, just to test mask path error
             dummy_source_array = np.array([[0,1e-6],[1,1e-6]])
             np.savetxt(valid_source, dummy_source_array, delimiter=',')

        # Expect FileNotFoundError from Raster or potentially IOError later
        with self.assertRaises((FileNotFoundError, IOError, OSError), msg="Should raise error for non-existent mask file"):
            Rainfall(rain_mask=non_existent_mask,
                     rain_source=valid_source,
                     source_sep=',',
                     dem_ras=self.base_dem_raster)

    def test_init_invalid_source_path(self):
        """Test error handling for non-existent source file path."""
        non_existent_source = os.path.join(self.test_dir, "non_existent_source.csv")
         # Use a valid mask path if available, otherwise skip or use dummy valid path concept
        valid_mask = getattr(self, 'sample_mask_path', 'dummy_valid_mask.asc')
        if not self.sample_data_available and valid_mask == 'dummy_valid_mask.asc':
             # Create dummy valid mask file if sample wasn't loaded, just to test source path error
             dummy_mask_array = np.zeros((self.base_dem_raster.shape))
             arcgridwrite(valid_mask, dummy_mask_array, self.base_dem_raster.header)

        # Expect FileNotFoundError or similar from np.loadtxt or internal check
        with self.assertRaises((FileNotFoundError, IOError, OSError), msg="Should raise error for non-existent source file"):
            Rainfall(rain_mask=valid_mask,
                     rain_source=non_existent_source,
                     source_sep=',',
                     dem_ras=self.base_dem_raster)

    def test_init_mismatched_mask_shape(self):
        """Test error handling for mismatched mask array shape vs DEM shape."""
        # Create intentionally wrong-sized mask array
        wrong_shape_mask = np.ones((self.base_dem_raster.shape[0] + 5, self.base_dem_raster.shape[1])) # Different rows
        # Use a valid source path if available
        valid_source = getattr(self, 'sample_source_path', 'dummy_valid_source.csv')
        if not self.sample_data_available and valid_source == 'dummy_valid_source.csv':
             dummy_source_array = np.array([[0,1e-6],[1,1e-6]])
             np.savetxt(valid_source, dummy_source_array, delimiter=',')


        # Expect ValueError from the check within Rainfall or its helpers
        with self.assertRaisesRegex(ValueError, "shape of rainfall_mask array is not consistent with DEM"):
            Rainfall(rain_mask=wrong_shape_mask,
                     rain_source=valid_source,
                     source_sep=',',
                     dem_ras=self.base_dem_raster)

    def test_init_invalid_source_type(self):
        """Test error handling for invalid data type for rain_source."""
         # Use a valid mask path if available
        valid_mask = getattr(self, 'sample_mask_path', 'dummy_valid_mask.asc')
        if not self.sample_data_available and valid_mask == 'dummy_valid_mask.asc':
             dummy_mask_array = np.zeros((self.base_dem_raster.shape))
             arcgridwrite(valid_mask, dummy_mask_array, self.base_dem_raster.header)

        # Create invalid rain source type (dictionary)
        invalid_source = {'time': [0, 3600], 'rate': [0.001, 0.002]}

        # Expect IOError from the check within set_source
        with self.assertRaisesRegex(IOError, "rain_source must be either a filename or a numpy array"):
            Rainfall(rain_mask=valid_mask,
                     rain_source=invalid_source,
                     dem_ras=self.base_dem_raster)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
