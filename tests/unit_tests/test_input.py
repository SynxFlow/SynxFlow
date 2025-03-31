import unittest
import numpy as np
import os
import shutil
from synxflow.IO import InputModel, Raster
from synxflow.IO.demo_functions import get_sample_data


class TestInputModelWithSampleData(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures, load sample data and create temp folders."""
        print("Setting up test with sample data...")
        # Get sample data paths and dictionary using the function from the tutorial
        self.dem_file, self.sample_data, self.data_path = get_sample_data() # [cite: 447, 597]

        # Load the actual sample DEM
        print(f"Loading sample DEM: {self.dem_file}")
        self.dem_raster = Raster(self.dem_file) # [cite: 606]
        self.rows, self.cols = self.dem_raster.shape
        self.header = self.dem_raster.header

        # Load sample rain mask (needed for rainfall test)
        self.rain_mask_file = os.path.join(self.data_path, 'rain_mask.gz')
        print(f"Loading sample rain mask: {self.rain_mask_file}")
        self.rain_mask_raster = Raster(self.rain_mask_file) # [cite: 608]

        # Load sample landcover (needed for landcover-based param test if added)
        self.landcover_file = os.path.join(self.data_path, 'landcover.gz')
        print(f"Loading sample landcover: {self.landcover_file}")
        self.landcover_raster = Raster(self.landcover_file) # [cite: 611]

        # Create a temporary directory for test outputs
        self.test_dir = "temp_test_input_model_sample"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        print(f"Temporary directory created: {self.test_dir}")

    def tearDown(self):
        """Tear down test fixtures, remove temporary directory."""
        print("Tearing down test...")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Temporary directory removed: {self.test_dir}")

    def test_initialization_with_sample_raster(self):
        """Test InputModel initialization using the sample Raster object."""
        model = InputModel(dem_data=self.dem_raster, case_folder=self.test_dir)
        self.assertIsInstance(model.DEM, Raster)
        self.assertEqual(model.shape, self.dem_raster.shape)
        self.assertEqual(model.header['ncols'], self.header['ncols'])
        self.assertEqual(model.num_of_sections, 1)
        self.assertEqual(model.get_case_folder(), self.test_dir)
        # Check default attributes
        self.assertEqual(model.attributes['manning'], 0.035) # Default Manning's n
        self.assertTrue(hasattr(model, 'Boundary'))
        self.assertTrue(hasattr(model, 'Rainfall'))

    def test_set_boundary_condition_with_sample(self):
        """Test setting boundary conditions using sample data."""
        model = InputModel(dem_data=self.dem_raster, case_folder=self.test_dir)
        # Use boundary_condition data from the loaded sample_data dict [cite: 447, 621]
        boundary_list_sample = self.sample_data['boundary_condition']
        model.set_boundary_condition(boundary_list=boundary_list_sample, outline_boundary='fall') # [cite: 621] Matches tutorial default

        self.assertEqual(model.Boundary.num_of_bound, 3) # 2 defined + 1 outline
        self.assertEqual(model.Boundary.data_table.type[0], 'fall') # Check outline type
        self.assertEqual(model.Boundary.data_table.type[1], 'open')
        self.assertEqual(model.Boundary.data_table.type[2], 'open')
        # Check if sources were assigned (using tutorial data)
        self.assertIsNotNone(model.Boundary.data_table.hUSources[1]) # Discharge upstream [cite: 448, 620]
        self.assertIsNotNone(model.Boundary.data_table.hSources[2]) # Depth downstream [cite: 449, 621]
        # Check discharge was converted to velocity (3 columns)
        self.assertEqual(model.Boundary.hU_sources[1].shape[1], 3)

    def test_set_rainfall_with_sample(self):
        """Test setting rainfall using sample data."""
        model = InputModel(dem_data=self.dem_raster, case_folder=self.test_dir)
        # Use rain_mask raster and rain_source from sample data [cite: 447, 608, 622]
        rain_source_sample = self.sample_data['rain_source'] # This is already a numpy array [cite: 447]
        model.set_rainfall(rain_mask=self.rain_mask_raster, rain_source=rain_source_sample) # [cite: 622]

        np.testing.assert_array_equal(model.Rainfall.get_source_array(), rain_source_sample)
        # Check mask was applied correctly (might need resampling internally)
        loaded_mask_array = model.Rainfall.get_mask_array()
        self.assertEqual(loaded_mask_array.shape, self.dem_raster.shape)

    def test_set_grid_parameter_landcover_with_sample(self):
        """Test setting grid parameters based on sample landcover."""
        model = InputModel(dem_data=self.dem_raster, case_folder=self.test_dir)
        # Set landcover first using the loaded sample raster [cite: 623]
        model.set_landcover(self.landcover_raster)
        self.assertTrue(hasattr(model, 'Landcover'))

        # Define parameters based on landcover as in the tutorial [cite: 623]
        manning_params = {'param_value': [0.035, 0.055],
                          'land_value': [0, 1],
                          'default_value': 0.035}
        model.set_grid_parameter(manning=manning_params) # [cite: 623]

        # Check if the manning attribute is now an array
        self.assertIsInstance(model.attributes['manning'], np.ndarray)
        self.assertEqual(model.attributes['manning'].shape, self.dem_raster.shape)
        # Verify some values based on landcover (requires knowing landcover values at specific points)
        # For example, if landcover at (row, col) is 0, manning should be 0.035
        # If landcover at (r2, c2) is 1, manning should be 0.055
        # This requires inspecting self.landcover_raster.array - skipping detailed check here for brevity

    def test_set_gauges_position_with_sample(self):
        """Test setting gauge positions using sample data."""
        model = InputModel(dem_data=self.dem_raster, case_folder=self.test_dir)
        gauges_pos_sample = self.sample_data['gauges_pos'] # [cite: 447, 451]
        model.set_gauges_position(gauges_pos_sample) # [cite: 624]
        np.testing.assert_array_equal(model.attributes['gauges_pos'], gauges_pos_sample)

    def test_write_input_files_structure_sample(self):
        """Test if core input files are created using sample setup."""
        model = InputModel(dem_data=self.dem_raster, case_folder=self.test_dir)
        # Apply sample settings before writing
        model.set_initial_condition('h0', 0.0) # [cite: 616]
        model.set_boundary_condition(boundary_list=self.sample_data['boundary_condition'], outline_boundary='fall') # [cite: 621]
        model.set_rainfall(rain_mask=self.rain_mask_raster, rain_source=self.sample_data['rain_source']) # [cite: 622]
        model.set_landcover(self.landcover_raster) # [cite: 623]
        model.set_grid_parameter(manning={'param_value': [0.035, 0.055], 'land_value': [0, 1], 'default_value':0.035}) # [cite: 623]
        model.set_gauges_position(self.sample_data['gauges_pos']) # [cite: 624]
        model.set_runtime([0, 7200, 900, 7200]) # [cite: 625]

        model.write_input_files() # [cite: 627]

        # Check if essential directories and files exist
        self.assertTrue(os.path.isdir(os.path.join(self.test_dir, 'input')))
        self.assertTrue(os.path.isdir(os.path.join(self.test_dir, 'input', 'mesh')))
        self.assertTrue(os.path.isdir(os.path.join(self.test_dir, 'input', 'field')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'mesh', 'DEM.txt')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'times_setup.dat')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'h.dat')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'hU.dat')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'manning.dat')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'precipitation_mask.dat')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'precipitation_source_all.dat')))
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'gauges_pos.dat')))
        # Check boundary files based on sample data (2 boundaries + outline, hU for #1, h for #2)
        # Boundary source file names are like h_BC_0.dat, hU_BC_0.dat etc. Index depends on order and if source exists.
        # Outline (fall) -> h_BC_0.dat, hU_BC_0.dat
        # Upstream (hU) -> hU_BC_1.dat
        # Downstream (h) -> h_BC_1.dat
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'h_BC_0.dat'))) # From outline=fall
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'hU_BC_0.dat'))) # From outline=fall
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'hU_BC_1.dat'))) # From upstream discharge
        self.assertTrue(os.path.isfile(os.path.join(self.test_dir, 'input', 'field', 'h_BC_1.dat'))) # From downstream depth

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)