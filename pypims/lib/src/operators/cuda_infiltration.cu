// ======================================================================================
// Name                :    High-Performance Integrated Modelling System
// Description         :    This code pack provides a generic framework for developing 
//                          Geophysical CFD software. Legacy name: GeoClasses
// ======================================================================================
// Version             :    0.2
// Author              :    Xilin Xia
// Create Time         :    2014/10/04
// Update Time         :    2020/04/26
// ======================================================================================
// LICENCE: GPLv3 
// ======================================================================================

/*!
\file cuda_infiltration.cu
\brief Source file for friction operator

\version 0.1
\author xilin xia

*/

#include "cuda_infiltration.h"
#include "cuda_kernel_launch_parameters.h"

namespace GC{

  namespace fv{

    __global__ void cuInfiltrationGreenAmptKernel(Scalar* h, Scalar* hydraulic_conductivity, Scalar* capillary_head, Scalar* water_content_diff, Scalar* culmulative_depth, Scalar delta_t, unsigned int size){
      unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
      while (index < size){
        Scalar small_value = 1e-10;
        Scalar _h = h[index];
        Scalar K_s = hydraulic_conductivity[index];
        Scalar phi_s = capillary_head[index];
        Scalar delta_theta = water_content_diff[index];
        Scalar F_0 = culmulative_depth[index];
        Scalar total_head = phi_s + _h;
        Scalar F_1 = 0.5*(F_0 + delta_t*K_s + sqrt((F_0 + delta_t*K_s)*(F_0 + delta_t*K_s)+4.0*delta_t*K_s*total_head*delta_theta));
        Scalar delta_F = fmin(_h,F_1 - F_0);
        culmulative_depth[index] += delta_F;
        h[index] -= delta_F;
        index += blockDim.x * gridDim.x;
      }

    }


    void cuInfiltrationGreenAmpt(cuFvMappedField<Scalar, on_cell>& h, cuFvMappedField<Scalar, on_cell>& hydraulic_conductivity, cuFvMappedField<Scalar, on_cell>& capillary_head, cuFvMappedField<Scalar, on_cell>& water_content_diff, cuFvMappedField<Scalar, on_cell>& culmulative_depth, Scalar delta_t){

      cuInfiltrationGreenAmptKernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(h.data.dev_ptr(), hydraulic_conductivity.data.dev_ptr(), capillary_head.data.dev_ptr(), water_content_diff.data.dev_ptr(), culmulative_depth.data.dev_ptr(), delta_t, h.data.size());

    }

    __global__ void cuTotalSourceSinkKernel(Scalar* h, Vector2* hU, Scalar* hydraulic_conductivity, Scalar* capillary_head, Scalar* water_content_diff, Scalar* culmulative_depth, Scalar* precipitation, Scalar* sewer_sink, Scalar delta_t, unsigned int size){
      unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;
      while (index < size){
        Scalar small_value = 1e-10;
        Scalar _h = h[index];
        Scalar _precipitation = precipitation[index];
        Scalar _sewer_sink = sewer_sink[index];
        Vector2 _hU = hU[index];
        Vector2 U = Vector2(0.0, 0.0); 
        if(_h > small_value){
          U = _hU/_h;
        }
        Scalar K_s = hydraulic_conductivity[index];
        Scalar phi_s = capillary_head[index];
        Scalar delta_theta = water_content_diff[index];
        Scalar F_0 = culmulative_depth[index];
        Scalar total_head = phi_s + _h;
        Scalar F_1 = 0.5*(F_0 + delta_t*K_s + sqrt((F_0 + delta_t*K_s)*(F_0 + delta_t*K_s)+4.0*delta_t*K_s*total_head*delta_theta));
        Scalar delta_F = fmin(_h,F_1 - F_0);
        culmulative_depth[index] += delta_F;
        Scalar h1 = _h - delta_F + (_precipitation - _sewer_sink)*delta_t;
        Scalar h2 = _h - delta_F - _sewer_sink*delta_t;
        h[index] = fmax(h1, 0.0);
        hU[index] = fmax(h2, 0.0)*U;
        index += blockDim.x * gridDim.x;
      }

    }


    void cuTotalSourceSink(cuFvMappedField<Scalar, on_cell>& h, cuFvMappedField<Vector, on_cell>& hU, cuFvMappedField<Scalar, on_cell>& hydraulic_conductivity, cuFvMappedField<Scalar, on_cell>& capillary_head, cuFvMappedField<Scalar, on_cell>& water_content_diff, cuFvMappedField<Scalar, on_cell>& culmulative_depth, cuFvMappedField<Scalar, on_cell>& precipitation, cuFvMappedField<Scalar, on_cell>& sewer_sink, Scalar delta_t){

      cuTotalSourceSinkKernel << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(h.data.dev_ptr(), hU.data.dev_ptr(), hydraulic_conductivity.data.dev_ptr(), capillary_head.data.dev_ptr(), water_content_diff.data.dev_ptr(), culmulative_depth.data.dev_ptr(), precipitation.data.dev_ptr(), sewer_sink.data.dev_ptr(), delta_t, h.data.size());

    }

  }

}