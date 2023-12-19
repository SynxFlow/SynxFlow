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
  m.doc() = "core engine for landslide runout model"; // optional module docstring

  m.def("run", &run, "landslide runout simulation with single gpu");
}
