import unittest
import numpy as np
import os
import shutil
import pandas as pd # Needed for comparing gauge results easily

from synxflow.IO import InputModel, OutputModel, Raster
from synxflow.IO.demo_functions import get_sample_data
from synxflow.IO.spatial_analysis import arcgridwrite # Need this helper

class TestOutputModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load base data once for all tests."""
        print("Setting up class: Loading sample data...")
        cls.dem_file, cls.sample_data_dict, cls.data_path = get_sample_data()
        cls.base_dem_raster = Raster(cls.dem_file)
        cls.base_header = cls.base_dem_raster.header
        cls.rows, cls.cols = cls.base_dem_raster.shape

        # Define data for fake output files consistent across tests
        cls.num_gauges = 2
        cls.gauge_pos = np.array([[560., 1030.], [1140., 330.]]) # Matches tutorial
        cls.times = np.array([0., 900., 1800.])
        cls.fake_h_values = np.array([[0.1, 0.2], [0.5, 0.6], [0.8, 0.9]]) # time x gauge
        cls.fake_hu_values_x = np.array([[0.01, 0.02], [0.05, 0.06], [0.08, 0.09]])
        cls.fake_hu_values_y = np.array([[0.001, 0.002], [0.005, 0.006], [0.008, 0.009]])
        cls.fake_eta_values = np.array([[10.1, 10.2], [10.5, 10.6], [10.8, 10.9]])
        cls.fake_h_max_array = np.random.rand(cls.rows, cls.cols) * 1.5 # Fake max depth grid

    def setUp(self):
        """Set up test fixtures, create dummy folders and fake output files."""
        print(f"\nSetting up test: {self.id()}")
        self.test_dir = f"temp_test_output_model_{self.id().split('.')[-1]}"
        self.input_dir = os.path.join(self.test_dir, 'input')
        self.output_dir = os.path.join(self.test_dir, 'output')
        self.mesh_dir = os.path.join(self.input_dir, 'mesh')
        self.field_dir = os.path.join(self.input_dir, 'field')

        os.makedirs(self.mesh_dir, exist_ok=True)
        os.makedirs(self.field_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Created directories under: {self.test_dir}")

        # --- Create Fake Input Files Needed by OutputModel ---
        # Create a minimal InputModel instance primarily for header/paths
        self.input_model = InputModel(dem_data=self.base_dem_raster, case_folder=self.test_dir)
        # Write fake gauges_pos.dat needed by read_gauges_file
        gauge_pos_path = os.path.join(self.field_dir, 'gauges_pos.dat')
        np.savetxt(gauge_pos_path, self.gauge_pos, fmt='%g')
        print(f"Created fake file: {gauge_pos_path}")

        # --- Create Fake Output Files ---
        # Fake h_gauges.dat
        h_gauge_path = os.path.join(self.output_dir, 'h_gauges.dat')
        h_gauge_data = np.hstack((self.times[:, np.newaxis], self.fake_h_values))
        np.savetxt(h_gauge_path, h_gauge_data, fmt='%g')
        print(f"Created fake file: {h_gauge_path}")

        # Fake hU_gauges.dat (interleaved hUx, hUy)
        hu_gauge_path = os.path.join(self.output_dir, 'hU_gauges.dat')
        hu_interleaved = np.empty((self.times.size, 1 + self.num_gauges * 2))
        hu_interleaved[:, 0] = self.times
        hu_interleaved[:, 1::2] = self.fake_hu_values_x
        hu_interleaved[:, 2::2] = self.fake_hu_values_y
        np.savetxt(hu_gauge_path, hu_interleaved, fmt='%g')
        print(f"Created fake file: {hu_gauge_path}")

        # Fake eta_gauges.dat
        eta_gauge_path = os.path.join(self.output_dir, 'eta_gauges.dat')
        eta_gauge_data = np.hstack((self.times[:, np.newaxis], self.fake_eta_values))
        np.savetxt(eta_gauge_path, eta_gauge_data, fmt='%g')
        print(f"Created fake file: {eta_gauge_path}")

        # Fake h_max_1800.asc (using last time step from fake gauge data)
        self.hmax_tag = f'h_max_{int(self.times[-1])}' # e.g., h_max_1800
        h_max_path = os.path.join(self.output_dir, f'{self.hmax_tag}.asc')
        # Use arcgridwrite helper with the base header and fake array data
        arcgridwrite(h_max_path, self.fake_h_max_array, self.base_header)
        print(f"Created fake file: {h_max_path}")


    def tearDown(self):
        """Tear down test fixtures, remove temporary directory."""
        print(f"Tearing down test: {self.id()}")
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            print(f"Removed directory: {self.test_dir}")

    def test_initialization_with_input_obj(self):
        """Test OutputModel initialization using an InputModel object."""
        output_model = OutputModel(input_obj=self.input_model) #
        self.assertEqual(output_model.case_folder, self.input_model.get_case_folder())
        self.assertEqual(output_model.num_of_sections, self.input_model.num_of_sections)
        self.assertEqual(output_model.header, self.input_model.header)
        self.assertEqual(output_model.input_folder, self.input_dir)
        self.assertEqual(output_model.output_folder, self.output_dir)
        self.assertTrue(hasattr(output_model, 'Summary'))

    def test_initialization_with_folder(self):
        """Test OutputModel initialization using folder path and sections."""
        # Needs a file for header detection, let's ensure DEM.txt exists from InputModel's write
        # Or create a dummy one if InputModel write is not assumed here. Let's create it.
        dem_txt_path = os.path.join(self.mesh_dir, 'DEM.txt')
        arcgridwrite(dem_txt_path, self.base_dem_raster.array, self.base_header)
        print(f"Created fake file for header detection: {dem_txt_path}")

        output_model = OutputModel(case_folder=self.test_dir, num_of_sections=1) #
        self.assertEqual(output_model.case_folder, self.test_dir)
        self.assertEqual(output_model.num_of_sections, 1)
        self.assertEqual(output_model.input_folder, self.input_dir)
        self.assertEqual(output_model.output_folder, self.output_dir)
        # Check if header was loaded correctly from the dummy DEM.txt
        self.assertIsNotNone(getattr(output_model, 'header', None))
        if hasattr(output_model, 'header'):
            self.assertEqual(output_model.header['ncols'], self.base_header['ncols'])
            self.assertEqual(output_model.header['nrows'], self.base_header['nrows'])

    def test_read_gauges_file_h(self):
        """Test reading the fake h_gauges.dat file."""
        output_model = OutputModel(input_obj=self.input_model)
        gauges_pos, times, values = output_model.read_gauges_file(file_tag='h') #

        np.testing.assert_array_equal(gauges_pos, self.gauge_pos)
        np.testing.assert_array_equal(times, self.times)
        self.assertEqual(values.shape, self.fake_h_values.shape)
        np.testing.assert_allclose(values, self.fake_h_values)
        # Check if data stored internally
        self.assertTrue('h' in output_model.gauge_values_all)
        np.testing.assert_allclose(output_model.gauge_values_all['h'], self.fake_h_values)

    def test_read_gauges_file_hu(self):
        """Test reading the fake hU_gauges.dat file."""
        output_model = OutputModel(input_obj=self.input_model)
        gauges_pos, times, values = output_model.read_gauges_file(file_tag='hU') #

        np.testing.assert_array_equal(gauges_pos, self.gauge_pos)
        np.testing.assert_array_equal(times, self.times)
        # values should be a list or tuple [values_x, values_y] based on OutputModel logic
        self.assertIsInstance(values, np.ndarray) # Assuming it returns numpy array [2, time, gauge]
        self.assertEqual(values.shape, (2, self.times.size, self.num_gauges)) # Check shape: (components, time, gauge)
        np.testing.assert_allclose(values[0], self.fake_hu_values_x) # Check x component
        np.testing.assert_allclose(values[1], self.fake_hu_values_y) # Check y component
        self.assertTrue('hU' in output_model.gauge_values_all)
        np.testing.assert_allclose(output_model.gauge_values_all['hU'][0], self.fake_hu_values_x)
        np.testing.assert_allclose(output_model.gauge_values_all['hU'][1], self.fake_hu_values_y)


    def test_read_gauges_file_eta(self):
        """Test reading the fake eta_gauges.dat file."""
        output_model = OutputModel(input_obj=self.input_model)
        gauges_pos, times, values = output_model.read_gauges_file(file_tag='eta') #

        np.testing.assert_array_equal(gauges_pos, self.gauge_pos)
        np.testing.assert_array_equal(times, self.times)
        self.assertEqual(values.shape, self.fake_eta_values.shape)
        np.testing.assert_allclose(values, self.fake_eta_values)
        self.assertTrue('eta' in output_model.gauge_values_all)
        np.testing.assert_allclose(output_model.gauge_values_all['eta'], self.fake_eta_values)


    def test_read_grid_file(self):
        """Test reading the fake .asc grid file."""
        output_model = OutputModel(input_obj=self.input_model)
        grid_obj = output_model.read_grid_file(file_tag=self.hmax_tag) # Use tag like 'h_max_1800'

        self.assertIsInstance(grid_obj, Raster)
        self.assertEqual(grid_obj.header['ncols'], self.base_header['ncols'])
        self.assertEqual(grid_obj.header['nrows'], self.base_header['nrows'])
        self.assertEqual(grid_obj.header['xllcorner'], self.base_header['xllcorner'])
        # Compare arrays, handling potential NaNs if arcgridwrite wrote NODATA
        expected_array = self.fake_h_max_array.copy()
        read_array = grid_obj.array.copy()
        # Assume NODATA was written as -9999 and read back as NaN
        expected_array[expected_array == self.base_header['NODATA_value']] = np.nan
        np.testing.assert_allclose(read_array, expected_array, rtol=1e-7, atol=1e-5, equal_nan=True)


# Add more tests, e.g., for multi-GPU file reading if needed (would require creating fake sub-folders '0', '1' etc.)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)