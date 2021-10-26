// ======================================================================================
// Name                :    GeoClasses : Generic Geophysical Flow Modelling Framework
// Description         :    This code pack provides a generic framework for developing 
//                          Geophysical CFD software.
// ======================================================================================
// Version             :    0.2 
// Author              :    Xilin Xia (PhD candidate in Newcastle University)
// Create Time         :    2014/10/04
// Update Time         :    2021/10/26
// ======================================================================================
// Copyright @ Xilin Xia 2021 . All rights reserved.
// ======================================================================================

/*!
\file cuda_infiltration.h
\brief Header file for cuda infiltration class

\version 0.1
\author xilin xia
*/

#ifndef CUDA_INFILTRATION_H
#define CUDA_INFILTRATION_H

#include "cuda_mapped_field.h"
#include "Scalar.h"
#include "Vector.h"
#include "Tensor.h"
#include "Flag.h"

namespace GC{

  namespace fv{

    void cuInfiltrationGreenAmpt(cuFvMappedField<Scalar, on_cell>& h, cuFvMappedField<Scalar, on_cell>& hydraulic_conductivity, cuFvMappedField<Scalar, on_cell>& capillary_head, cuFvMappedField<Scalar, on_cell>& water_content_diff, cuFvMappedField<Scalar, on_cell>& culmulative_depth, Scalar delta_t);

    // combining infiltration, precipitation and sink - to add evapotransporation
    void cuTotalSourceSink(cuFvMappedField<Scalar, on_cell>& h, cuFvMappedField<Vector, on_cell>& hU, cuFvMappedField<Scalar, on_cell>& hydraulic_conductivity, cuFvMappedField<Scalar, on_cell>& capillary_head, cuFvMappedField<Scalar, on_cell>& water_content_diff, cuFvMappedField<Scalar, on_cell>& culmulative_depth, cuFvMappedField<Scalar, on_cell>& precipitation, cuFvMappedField<Scalar, on_cell>& sewer_sink, Scalar delta_t);

  }

}

#endif