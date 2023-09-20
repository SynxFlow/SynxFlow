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
  m.doc() = "core engine for debris flow model"; // optional module docstring

  m.def("run", &run, "debris flow simulation with single gpu");
}
