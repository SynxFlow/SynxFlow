// ======================================================================================
// Name                :    High-Performance Integrated Modelling System
// Description         :    This code pack provides a generic framework for developing 
//                          Geophysical CFD software. Legacy name: GeoClasses
// ======================================================================================
// Version             :    1.0.1 
// Author              :    Xilin Xia
// Create Time         :    2014/10/04
// Update Time         :    2020/04/26
// ======================================================================================
// LICENCE: GPLv3 
// ======================================================================================


/*!
\file urban_flood_simulator.cu
\brief Source file for component test

*/

#include "cuda_flood_solvers.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(flood, m) {
    m.doc() = R"pbdoc(
        Flood Simulation Engine
        ==============================

        This module provides the core functionality for running a flood simulation.
    )pbdoc";

    m.def("run", &run, R"pbdoc(
        run(work_dir)

        Executes a flood simulation on a single GPU.

        Parameters:
            work_dir (str): The path to the working directory that contains all
                necessary input files and configuration data for the simulation.

        Returns:
            int: A status code where 0 indicates that the simulation completed
                successfully, and any non-zero value indicates an error.

    )pbdoc");

    m.def("run_mgpus", &run_mgpus, R"pbdoc(
        run_mgpus(work_dir)

        Executes a flood simulation utilizing multiple GPUs.

        Parameters:
            work_dir (str): The path to the working directory with the necessary
                input data and configuration files.

        Returns:
            int: A status code indicating the outcome of the simulation (0 for success,
                non-zero for failure).
    )pbdoc");
}

