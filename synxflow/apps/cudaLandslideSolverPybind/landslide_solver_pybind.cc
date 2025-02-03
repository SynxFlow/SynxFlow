// ====================================================================================== 
// Author              :    Xilin Xia, University of Birmingham, x.xia.1@bham.ac.uk
// Update Time         :    2023/10/18
// ======================================================================================
// LICENCE: GPLv3 
// ======================================================================================



/*!
\file urban_flood_simulator.cu
\brief Source file for component test

*/

#include "cuda_landslide_solver.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(landslide, m) {
    m.doc() = R"pbdoc(
        Landslide Runout Simulation Engine
        ====================================

        This module provides the core functionality for running a landslide runout simulation.
    )pbdoc";

    m.def("run", &run, R"pbdoc(
        run(work_dir)

        Executes a landslide runout simulation using a single GPU.

        Parameters:
            work_dir (str): The path to the working directory that contains all the necessary
                input data and configuration files required for the simulation.

        Returns:
            int: A status code indicating the result of the simulation. A value of 0 typically
                means that the simulation completed successfully, while any non-zero value indicates
                that an error occurred.

    )pbdoc");
}

