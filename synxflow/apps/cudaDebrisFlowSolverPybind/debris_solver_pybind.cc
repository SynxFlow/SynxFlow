// ====================================================================================== 
// Author              :    Xilin Xia, University of Birmingham, x.xia.1@bham.ac.uk
// Update Time         :    2023/09/19
// ======================================================================================
// LICENCE: GPLv3 
// ======================================================================================



/*!
\file urban_flood_simulator.cu
\brief Source file for component test

*/

#include "cuda_debris_flow_solver.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(debris, m) {
  m.doc() = R"pbdoc(
        Debris Flow Simulation Engine
        ==============================

        This module provides the core functionality for running a debris flow simulation.
    )pbdoc"; // optional module docstring

  m.def("run", &run, R"pbdoc(
        run(work_dir)

        Runs the debris flow simulation using the specified working directory.

        Parameters:
            work_dir (str): A path to the working directory where all necessary input
                files (such as configuration or data files) are located. This directory
                is also used to store the simulation outputs.

        Returns:
            int: A status code indicating the result of the simulation. A value of 0 typically
                means the simulation completed successfully, while a non-zero value indicates an error.

        Note:
            Ensure that the working directory is correctly set up with the required input data.
    )pbdoc");
}
