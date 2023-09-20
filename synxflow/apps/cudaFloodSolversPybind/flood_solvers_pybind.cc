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
  m.doc() = "core engine for hipims model"; // optional module docstring

  m.def("run", &run, "flood simulation with single gpu");
  m.def("run_mgpus", &run_mgpus, "flood simulation with multiple gpus");
}
