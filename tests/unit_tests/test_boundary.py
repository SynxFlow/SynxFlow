import unittest
import numpy as np
import pandas as pd

from synxflow.IO.Boundary import Boundary

class TestBoundary(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.times = np.array([[0.], [100.], [200.]])
        self.dummy_h = np.hstack((self.times, self.times * 0.01 + 5.0)) # time, depth
        self.dummy_q = np.hstack((self.times, self.times * -0.1 + 20.0)) # time, discharge
        self.dummy_v = np.hstack((self.times, self.times * 0.001, self.times * -0.0005 + 0.1)) # time, vx, vy
        self.dummy_poly = np.array([[10., 10.], [20., 10.], [20., 20.], [10., 20.]])

        # Sample boundary list definitions
        self.boundary_list_simple = [
            {'polyPoints': self.dummy_poly, 'type': 'open', 'h': self.dummy_h, 'name': 'Downstream'}
        ]
        self.boundary_list_mixed = [
            {'polyPoints': self.dummy_poly, 'type': 'open', 'h': self.dummy_h, 'name': 'Outlet H'},
            {'polyPoints': self.dummy_poly + 50, 'type': 'open', 'hU': self.dummy_q, 'name': 'Inlet Q'},
            {'polyPoints': self.dummy_poly + 100, 'type': 'open', 'hU': self.dummy_v, 'name': 'Inlet V'},
            {'polyPoints': self.dummy_poly + 150, 'type': 'rigid', 'name': 'Wall'},
            # A boundary without polyPoints should modify the outline
            {'type': 'open', 'h': self.dummy_h, 'name': 'Outline H'}
        ]

        # Dummy DEM header needed for methods like _fetch_boundary_cells, _convert_flow2velocity
        # Create only if testing those methods specifically
        self.dummy_header = {'ncols': 50, 'nrows': 50, 'xllcorner': 0, 'yllcorner': 0, 'cellsize': 10, 'NODATA_value': -9999}
        # Dummy subs - very simplified, just for existence checks perhaps
        self.dummy_valid_subs = (np.arange(10), np.arange(10)) # Placeholder
        self.dummy_outline_subs = (np.array([0, 1, 2]), np.array([0, 0, 0])) # Placeholder


    def test_init_default_fall(self):
        """Test initialization with default outline ('fall')."""
        boundary = Boundary()
        self.assertEqual(boundary.num_of_bound, 1)
        self.assertEqual(boundary.outline_boundary, 'fall')
        self.assertEqual(boundary.data_table.shape[0], 1)
        self.assertEqual(boundary.data_table['type'][0], 'fall')
        self.assertIsNone(boundary.data_table['extent'][0])
        self.assertIsNotNone(boundary.data_table['hSources'][0]) # Fall has default zero sources
        self.assertIsNotNone(boundary.data_table['hUSources'][0])
        np.testing.assert_array_equal(boundary.data_table['h_code'][0], np.array([[3, 0, 0]]))
        np.testing.assert_array_equal(boundary.data_table['hU_code'][0], np.array([[3, 0, 0]]))

    def test_init_outline_rigid(self):
        """Test initialization with 'rigid' outline."""
        boundary = Boundary(outline_boundary='rigid')
        self.assertEqual(boundary.num_of_bound, 1)
        self.assertEqual(boundary.outline_boundary, 'rigid')
        self.assertEqual(boundary.data_table['type'][0], 'rigid')
        self.assertIsNone(boundary.data_table['hSources'][0])
        self.assertIsNone(boundary.data_table['hUSources'][0])
        np.testing.assert_array_equal(boundary.data_table['h_code'][0], np.array([[2, 0, 0]]))
        np.testing.assert_array_equal(boundary.data_table['hU_code'][0], np.array([[2, 2, 0]]))

    def test_init_outline_open(self):
        """Test initialization with 'open' outline."""
        boundary = Boundary(outline_boundary='open')
        self.assertEqual(boundary.num_of_bound, 1)
        self.assertEqual(boundary.outline_boundary, 'open')
        self.assertEqual(boundary.data_table['type'][0], 'open')
        self.assertIsNone(boundary.data_table['hSources'][0])
        self.assertIsNone(boundary.data_table['hUSources'][0])
        np.testing.assert_array_equal(boundary.data_table['h_code'][0], np.array([[2, 0, 0]]))
        np.testing.assert_array_equal(boundary.data_table['hU_code'][0], np.array([[2, 1, 0]]))

    def test_init_with_simple_list(self):
        """Test initialization with a simple boundary list."""
        boundary = Boundary(boundary_list=self.boundary_list_simple, outline_boundary='rigid')
        self.assertEqual(boundary.num_of_bound, 2) # 1 outline + 1 defined
        self.assertEqual(boundary.data_table['type'][0], 'rigid')
        self.assertEqual(boundary.data_table['type'][1], 'open')
        self.assertEqual(boundary.data_table['name'][1], 'Downstream')
        np.testing.assert_array_equal(boundary.data_table['extent'][1], self.dummy_poly)
        np.testing.assert_array_equal(boundary.data_table['hSources'][1], self.dummy_h)
        self.assertIsNone(boundary.data_table['hUSources'][1])
        # Check codes (outline rigid, defined open+h)
        np.testing.assert_array_equal(boundary.data_table['h_code'][0], np.array([[2, 0, 0]]))
        np.testing.assert_array_equal(boundary.data_table['hU_code'][0], np.array([[2, 2, 0]]))
        np.testing.assert_array_equal(boundary.data_table['h_code'][1], np.array([[3, 0, 0]])) # h source index 0
        np.testing.assert_array_equal(boundary.data_table['hU_code'][1], np.array([[2, 1, 0]])) # Default open hU

    def test_init_with_mixed_list_and_outline_override(self):
        """Test initialization with mixed types and overriding outline."""
        # self.boundary_list_mixed contains an entry without polyPoints, which should override the outline_boundary='fall'
        boundary = Boundary(boundary_list=self.boundary_list_mixed, outline_boundary='fall')
        self.assertEqual(boundary.num_of_bound, 5) # 1 outline (overridden) + 4 defined with polyPoints

        # Check overridden outline (index 0, uses last entry in list without polyPoints)
        self.assertEqual(boundary.data_table['type'][0], 'open')
        self.assertEqual(boundary.data_table['name'][0], 'Outline H')
        self.assertIsNone(boundary.data_table['extent'][0])
        np.testing.assert_array_equal(boundary.data_table['hSources'][0], self.dummy_h)
        np.testing.assert_array_equal(boundary.data_table['h_code'][0], np.array([[3, 0, 0]])) # h source index 0
        np.testing.assert_array_equal(boundary.data_table['hU_code'][0], np.array([[2, 1, 0]])) # Default open hU

        # Check other defined boundaries by name/type
        # Outlet H (index 1)
        self.assertEqual(boundary.data_table['name'][1], 'Outlet H')
        self.assertEqual(boundary.data_table['type'][1], 'open')
        np.testing.assert_array_equal(boundary.data_table['hSources'][1], self.dummy_h)
        self.assertIsNone(boundary.data_table['hUSources'][1])
        np.testing.assert_array_equal(boundary.data_table['h_code'][1], np.array([[3, 0, 1]])) # h source index 1
        np.testing.assert_array_equal(boundary.data_table['hU_code'][1], np.array([[2, 1, 0]]))

        # Inlet Q (index 2)
        self.assertEqual(boundary.data_table['name'][2], 'Inlet Q')
        self.assertEqual(boundary.data_table['type'][2], 'open')
        self.assertIsNone(boundary.data_table['hSources'][2])
        np.testing.assert_array_equal(boundary.data_table['hUSources'][2], self.dummy_q)
        np.testing.assert_array_equal(boundary.data_table['h_code'][2], np.array([[2, 0, 0]]))
        np.testing.assert_array_equal(boundary.data_table['hU_code'][2], np.array([[3, 0, 0]])) # hU source index 0

        # Inlet V (index 3)
        self.assertEqual(boundary.data_table['name'][3], 'Inlet V')
        self.assertEqual(boundary.data_table['type'][3], 'open')
        self.assertIsNone(boundary.data_table['hSources'][3])
        np.testing.assert_array_equal(boundary.data_table['hUSources'][3], self.dummy_v)
        np.testing.assert_array_equal(boundary.data_table['h_code'][3], np.array([[2, 0, 0]]))
        np.testing.assert_array_equal(boundary.data_table['hU_code'][3], np.array([[3, 0, 1]])) # hU source index 1 (since Inlet Q was 0)

        # Wall (index 4)
        self.assertEqual(boundary.data_table['name'][4], 'Wall')
        self.assertEqual(boundary.data_table['type'][4], 'rigid')
        self.assertIsNone(boundary.data_table['hSources'][4])
        self.assertIsNone(boundary.data_table['hUSources'][4])
        np.testing.assert_array_equal(boundary.data_table['h_code'][4], np.array([[2, 0, 0]]))
        np.testing.assert_array_equal(boundary.data_table['hU_code'][4], np.array([[2, 2, 0]]))

    def test_get_summary_no_fetch(self):
        """Test get_summary before cell fetching (no cell counts)."""
        boundary = Boundary(boundary_list=self.boundary_list_simple, outline_boundary='rigid')
        summary = boundary.get_summary()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary['Number of boundaries'], '2')
        # 'Boundary details' might be empty or incomplete before _fetch_boundary_cells
        self.assertTrue('Boundary details' in summary)
        if summary['Boundary details']: # If it adds description even without counts
             self.assertIn('rigid', summary['Boundary details'][0])
             self.assertIn('open', summary['Boundary details'][1])
             self.assertNotIn('number of cells', summary['Boundary details'][0])


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)