// ====================================================================================== 
// Author              :    Xilin Xia, University of Birmingham, x.xia.1@bham.ac.uk
// Update Time         :    2024/08/30
// ======================================================================================
// LICENCE: GPLv3 
// ======================================================================================


/*!
\file cuda_debris_flow_solver.cu
\brief Source file for Debris Flow solver

*/
#ifdef _WIN32
#include <direct.h>
// MSDN recommends against using getcwd & chdir names
#define cwd _getcwd
#define cd _chdir
#else
#include "unistd.h"
#define cwd getcwd
#define cd chdir
#endif

#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

//These header files are the primitive types
#include "Flag.h"
#include "Scalar.h"
#include "Vector.h"
#include "cuda_arrays.h"
//These header files are for the fields
#include "mapped_field.h"
#include "cuda_mapped_field.h"
//These header files are for finite volume mesh
#include "mesh_fv_reduced.h"
#include "mesh_fv_cartesian.h"
#include "mesh_interface.h"
#include "cuda_mesh_fv.h"
//These header files are for input and output
#include "gisAsciiMesh_reader.h"
#include "gmsh_reader.h"
#include "field_reader.h"
#include "cuda_simple_writer.h"
#include "cuda_backup_writer.h"
#include "cuda_gauges_writer.h"
#include "cuda_gisascii_writer.h"
//These header files are for shallow water equations advection
#include "cuda_advection_NSWEs.h"
//These header files are for shallow water equations advection with transport
#include "cuda_transport_NSWEs.h"
//erosion and deposition rate
#include "cuda_erosion_deposition.h"
//The header file for gradient
#include "cuda_gradient.h"
//The header file for limiter
#include "cuda_limiter.h"
//The header file for friction
#include "cuda_friction.h"
//The header file for field algebra
#include "cuda_field_algebra.h"
//The header file for integrator
#include "cuda_integrators.h"
//The header file for device query
#include "cuda_device_query.h"
//The header file for time controllinh
#include "cuda_adaptive_time_control.h"
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

//using the name space for GeoClasses
using namespace GC;

int run(const char* work_dir){

  if(cd(work_dir) == -1){
    printf("The working directory does not exist!/n");
  }

  std::ifstream device_setup_file("input/device_setup.dat");
  int device_id;
  if (device_setup_file.is_open()){
    device_setup_file >> device_id;
	  checkCuda(cudaSetDevice(device_id));
	  std::cout << "GPU " << device_id << " is choosen as the model device"<< std::endl;
  }
  else{
    deviceQuery();
  }

  Scalar dt_out = 0.5;
  Scalar backup_interval = 0.0;
  Scalar backup_time = 0.0;
  Scalar t_current = 0.0;
  Scalar t_out = 0.0;
  Scalar t_all = 0.0;
  Scalar t_small = 1e-10;

  //*******************Read times setup value from file
  std::ifstream times_setup_file("input/times_setup.dat");
  if (!times_setup_file) {
    std::cout << "Please input current time, total time, output time interval and backup interval" << std::endl;
    std::cin >> t_current >> t_all >> dt_out >> backup_interval;
  }
  else {
    Scalar _time;
    std::vector<Scalar> GPU_Time_Values;
    while (times_setup_file >> _time) {
      GPU_Time_Values.push_back(_time);
    }
    t_current = GPU_Time_Values[0];
    t_all = GPU_Time_Values[1];
    dt_out = GPU_Time_Values[2];
    backup_interval = GPU_Time_Values[3];
    std::cout << "Current time: " << t_current << "s" << std::endl;
    std::cout << "Total time: " << t_all << "s" << std::endl;
    std::cout << "Output time interval: " << dt_out << "s" << std::endl;
    std::cout << "Backup interval: " << backup_interval << "s" << std::endl;
  }
 //********************************
 //*******************Read device setup value from file

  Scalar rho_water = 1000;
  Scalar rho_solid = 1580;
  Scalar dim_mid = 0.0039;
  Scalar porosity = 0.42;
  Scalar critical_slope = 0.7;
  Scalar alpha = 1.0;
  Scalar beta = 1.0;
  //******************Read parametres from file
  std::ifstream parameters_file("input/parameters.dat");
  if (!parameters_file) {
    std::cout << "Please input water density (kg/m3), solid density (kg/m3), particle diameter (m), porosity, critical slope, alpha and beta" << std::endl;
    std::cin >> rho_water >> rho_solid >> dim_mid >> porosity >> critical_slope >> alpha >> beta;
  }
  else{
    Scalar _parameter;
    std::vector<Scalar> Parameter_Values;
    while (parameters_file >> _parameter) {
      Parameter_Values.push_back(_parameter);
    }
    rho_water = Parameter_Values[0];
    rho_solid = Parameter_Values[1];
    dim_mid = Parameter_Values[2];
    porosity = Parameter_Values[3];
    critical_slope = Parameter_Values[4];
    alpha = Parameter_Values[5];
    beta = Parameter_Values[6];
  }


  //*******************************
  //******************Read parameters from file

  std::shared_ptr<unstructuredFvMesh>  mesh = std::make_shared<CartesianFvMesh>("input/mesh/DEM.txt");

  std::cout << "Read in mesh successfully" << std::endl;

  //creating mesh on device
  std::shared_ptr<cuUnstructuredFvMesh>  mesh_ptr_dev = std::make_shared<cuUnstructuredFvMesh>(fvMeshQueries(mesh));

  //Read in field data
  fvScalarFieldOnCell z_host(fvMeshQueries(mesh), completeFieldReader("input/field/", "z"));
  fvScalarFieldOnCell h_host(fvMeshQueries(mesh), completeFieldReader("input/field/", "h"));
  fvVectorFieldOnCell hU_host(fvMeshQueries(mesh), completeFieldReader("input/field/", "hU"));
  fvScalarFieldOnCell C_host(fvMeshQueries(mesh), completeFieldReader("input/field/", "C"));
  fvScalarFieldOnCell erodible_depth_host(fvMeshQueries(mesh), completeFieldReader("input/field/", "erodible_depth"));
  fvScalarFieldOnCell manning_coeff_host(fvMeshQueries(mesh), completeFieldReader("input/field/", "manning"));
  fvScalarFieldOnCell mu_dynamic_host(fvMeshQueries(mesh), completeFieldReader("input/field/", "dynamic_friction_coeff"));
  fvScalarFieldOnCell mu_static_host(fvMeshQueries(mesh), completeFieldReader("input/field/", "static_friction_coeff"));

  //precipitation
  fvScalarFieldOnCell precipitation_host(fvMeshQueries(mesh), completeFieldReader("input/field/", "precipitation"));

  std::cout << "Read in field successfully" << std::endl;

  //h, z, hU, C
  cuFvMappedField<Scalar, on_cell> z_old(z_host, mesh_ptr_dev);
  cuFvMappedField<Scalar, on_cell> z(z_host, mesh_ptr_dev);
  cuFvMappedField<Scalar, on_cell> h(h_host, mesh_ptr_dev);
  cuFvMappedField<Vector, on_cell> hU(hU_host, mesh_ptr_dev);
  cuFvMappedField<Scalar, on_cell> C(C_host, mesh_ptr_dev);
  cuFvMappedField<Scalar, on_cell> erodible_depth(erodible_depth_host, mesh_ptr_dev);
  cuFvMappedField<Scalar, on_cell> manning_coeff(manning_coeff_host, mesh_ptr_dev);
  cuFvMappedField<Scalar, on_cell> mu_dynamic(mu_dynamic_host, mesh_ptr_dev);
  cuFvMappedField<Scalar, on_cell> mu_static(mu_static_host, mesh_ptr_dev);

  //precipitation
  cuFvMappedField<Scalar, on_cell> precipitation(precipitation_host, mesh_ptr_dev);

  //new topography after collapse
  cuFvMappedField<Scalar, on_cell> z_new(h, partial);

  //Velocity
  cuFvMappedField<Vector, on_cell> u(hU, partial);

  //x and y components of hU
  cuFvMappedField<Scalar, on_cell> hUx(h, partial);
  cuFvMappedField<Scalar, on_cell> hUy(h, partial);

  //Solid phase depth
  cuFvMappedField<Scalar, on_cell> hC(h, partial);

  //concentration gradient
  cuFvMappedField<Vector, on_cell> C_grad(hU, partial);

  //creating gauges writer
  cuGaugesWriter<Scalar, on_cell> h_writer(fvMeshQueries(mesh), h, "input/field/gauges_pos.dat", "output/h_gauges.dat");
  cuGaugesWriter<Vector, on_cell> hU_writer(fvMeshQueries(mesh), hU, "input/field/gauges_pos.dat", "output/hU_gauges.dat");
  cuGaugesWriter<Scalar, on_cell> C_writer(fvMeshQueries(mesh), C, "input/field/gauges_pos.dat", "output/C_gauges.dat");

  //----Below is not needed------
  //find minimum z
  //Scalar min_z = thrust::reduce(thrust::device_ptr <Scalar>(z.data.dev_ptr()), thrust::device_ptr <Scalar>(z.data.dev_ptr() + z.data.size()), (Scalar) 3e35, thrust::minimum<Scalar>());

  //fv::cuUnaryOn(z, [=] __device__(Scalar& a) -> Scalar{ return a - min_z + 0.0001; });
  //-----------------------------
  //advections
  cuFvMappedField<Scalar, on_cell> h_advection(h, partial);
  cuFvMappedField<Vector, on_cell> hU_advection(hU, partial);
  cuFvMappedField<Scalar, on_cell> hC_advection(hC, partial);

  //friction force
  cuFvMappedField<Vector, on_cell> friction_force(hU, partial);

  //erosion and deposition rate
  cuFvMappedField<Scalar, on_cell> ED_rate(h, partial);

  //erosion and deposition momentum correction
  cuFvMappedField<Vector, on_cell> mom_correction(hU, partial);

  //gradient
  cuFvMappedField<Vector, on_cell> z_gradient(hU, partial);
  fv::cuLimitedGradientCartesian(z, z_gradient);

  //mass positivity time step constrain
//  cuFvMappedField<Scalar, on_cell> dt_mass(h, partial);

  //gravity
  cuFvMappedField<Scalar, on_cell> gravity(h, partial);
  //setting gravity to single value 9.81
  fv::cuUnaryOn(gravity, [] __device__(Scalar& a) -> Scalar{ return 9.81; });
  //fv::cuBinary(z_gradient, gravity, gravity, [] __device__(Vector& a, Scalar& b) -> Scalar{ return b / (1.0 + dot(a, a)); });

  t_out = t_current + dt_out;
  backup_time = t_current + backup_interval;

  cuAdaptiveTimeControl2D time_controller(0.005, t_all, 0.5, t_current);

  //write the initial profile
  //std::cout << "Writing initial conditions as output" << std::endl;
  //cuSimpleWriterLowPrecision(z, "z", time_controller.current());
  //cuSimpleWriterLowPrecision(h, "h", time_controller.current());
  //cuSimpleWriterLowPrecision(hU, "hU", time_controller.current());
  //cuSimpleWriterLowPrecision(hC, "hC", time_controller.current());

  auto filter = [] __device__(Vector& a, Scalar& b) ->Vector{
    if (b <= 1e-10){
      return Vector(0.0);
    }
    else{
      return a;
    }
  };

  auto filter2 = [] __device__(Scalar& a, Scalar& b) ->Scalar{
    if (a <= 1e-10){
      return 0.0;
    }
    else if(a >= b){
      return b;
    }else{
      return a;
    }
  };

  auto filter1 = [] __device__(Scalar& a) ->Scalar{
    if (a <= 1e-10){
      return 0.0;
    }
    else{
      return a;
    }
  };

  //calculate the velocity
  auto divide = [] __device__(Vector& a, Scalar& b) ->Vector{
    if (b >= 1e-10){
      return a / b;
    }
    else{
      return Vector(0.0);
    }
  };

  auto divide_scalar = [] __device__(Scalar& a, Scalar& b) ->Scalar{
    if (b >= 1e-10){
      return a / b;
    }
    else{
      return Scalar(0.0);
    }
  };

  auto multiply_scalar = [] __device__(Scalar& a, Scalar& b) ->Scalar{
    return a*b;
  };  

  //print current time
  std::cout << time_controller.current() << std::endl;

  z.update_time(time_controller.current(), 0.0);
  z.update_boundary_values();
  h.update_time(time_controller.current(), 0.0);
  h.update_boundary_values();
  hU.update_time(time_controller.current(), 0.0);
  hU.update_boundary_values();
  C.update_time(time_controller.current(), 0.0);
  C.update_boundary_values();
  
  //hC for the initial step
  fv::cuBinary(h, C, hC, multiply_scalar);

  //ascii raster writer
  cuGisAsciiWriter raster_writer("input/mesh/DEM.txt");

  //write initial depth
  raster_writer.write(h, "h", time_controller.current());

  std::ofstream fout;
  fout.open("output/timestep_log.txt");

  double total_runtime = 0.0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  int cnt = 0;
  //Main loop
  do{

    cudaEventRecord(start);

    Scalar dt = time_controller.dt();

    //calculate the surface elevation
    fv::cuBinary(hU, h, u, divide);

    //calculate advection
    fv::cuTransportNSWEsSRMCartesian(gravity, h, z, z_gradient, hU, hC, h_advection, hU_advection, hC_advection);

    //multiply advection with -1
    fv::cuUnaryOn(h_advection, [] __device__(Scalar& a) -> Scalar{ return -1.0*a; });
    fv::cuUnaryOn(hU_advection, [] __device__(Vector& a) -> Vector{ return -1.0*a; });
    fv::cuUnaryOn(hC_advection, [] __device__(Scalar& a) -> Scalar{ return -1.0*a; });

    //integration
    fv::cuEulerIntegrator(h, h_advection, time_controller.dt(), time_controller.current());
    fv::cuEulerIntegrator(hC, hC_advection, time_controller.dt(), time_controller.current());
    //fv::cuFrictionManningMCImplicit(time_controller.dt(), porosity, rho_water, rho_solid, gravity, manning_coeff, mu_dynamic, h, hC, hU, hU_advection);
    fv::cuFrictionManningImplicit(time_controller.dt(), gravity, manning_coeff, h, hU, hU_advection);

    //calculating erosion deposition rate
    //fv::cuEDMeyerPeterMuller(time_controller.dt(), gravity, h, hU, hC, manning_coeff, ED_rate, rho_solid, rho_water, dim_mid);
    fv::cuEDTakahashiIversonXia(gravity, h, hU, hC, manning_coeff, mu_dynamic, mu_static, ED_rate, rho_solid, rho_water, porosity, alpha, beta, dt, dim_mid);

    hU.update_time(time_controller.current(), time_controller.dt());
    hU.update_boundary_values();

    //constraining erosion deposition rate
    fv::cuBinary(ED_rate, hC, ED_rate, [=] __device__(Scalar& a, Scalar& b) -> Scalar{ return fmax(a, (Scalar)-1.0*b / dt); });
    //fv::cuBinary(ED_rate, hC, ED_rate, [=] __device__(Scalar& a, Scalar& b) -> Scalar { return fmax(a, 0.0); });
    fv::cuBinary(ED_rate, erodible_depth, ED_rate, [=] __device__(Scalar& a, Scalar& b) -> Scalar{ return fmin(a, b*((Scalar)1.0 - porosity) / dt); });
    
    //calculate the concentration
    fv::cuBinary(hC, h, C, divide_scalar);

    //calculating the concentration gradient
    fv::cuLimitedGradientCartesian(C, C_grad);

    //calculating momentum correction term
    fv::cuMomentumCorrection(gravity, h, hC, C_grad, hU, ED_rate, mom_correction, rho_solid, rho_water, porosity);

    //updating momentum
    fv::cuEulerIntegrator(hU, mom_correction, time_controller.dt(), time_controller.current());

    //updating solid phase depth
    fv::cuEulerIntegrator(hC, ED_rate, time_controller.dt(), time_controller.current());

    fv::cuUnaryOn(ED_rate, [=] __device__(Scalar& a) -> Scalar{ return a / (1.0 - porosity); });

    //updating depth
    fv::cuEulerIntegrator(h, ED_rate, time_controller.dt(), time_controller.current());

    //updating bed 
    fv::cuEulerIntegrator(z, ED_rate, -1.0*time_controller.dt(), time_controller.current());

    //updating erodible depth 
    fv::cuEulerIntegrator(erodible_depth, ED_rate, -1.0*time_controller.dt(), time_controller.current());

    //bank collpse
    //  if (cnt % 100 == 0){
    //   for (int i = 0; i < 10; i++){
    //     fv::cuBankCollapse(z, z_new, critical_slope);
    //     fv::cuUnary(z_new, z, [=] __device__(Scalar& a) -> Scalar{ return a; });
    //   }
    // }

    //precipitation
    precipitation.update_time(time_controller.current(), 0.0);
    precipitation.update_data_values();
    fv::cuEulerIntegrator(h, precipitation, time_controller.dt(), time_controller.current());

    //calculate the concentration, update the boundary, and then calculate hC again, so that concentration boundary condition can be applied
    fv::cuBinary(hC, h, C, divide_scalar);
    C.update_time(time_controller.current(), time_controller.dt());
    C.update_boundary_values();
    fv::cuBinary(h, C, hC, multiply_scalar);


    //forwarding the time
    time_controller.forward();
    time_controller.updateByCFL(gravity, h, hU);
// xiaodong add this to reduce the size of output gauge files
    if (cnt % 100 == 0){
        h_writer.write(time_controller.current());
        hU_writer.write(time_controller.current());
        C_writer.write(time_controller.current());
    }
//
    //h_writer.write(time_controller.current());
    //hU_writer.write(time_controller.current());

    //print current time
    printf("%f\n", time_controller.current());
    fout << time_controller.current() << " " << time_controller.dt() << std::endl;
    cnt++;

    fv::cuBinaryOn(hU, h, filter);
    //fv::cuUnaryOn(h, filter1);
    //fv::cuBinaryOn(hC, h, filter2);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time = 0.0;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    total_runtime += elapsed_time;

    if (time_controller.current() >= t_out - t_small){
      std::cout << "Writing output files" << std::endl;
      raster_writer.write(z, "z", t_out);
      raster_writer.write(h, "h", t_out);
      fv::cuUnary(hU, hUx, [] __device__(Vector& a) -> Scalar{ return a.x; });
      fv::cuUnary(hU, hUy, [] __device__(Vector& a) -> Scalar{ return a.y; });
      raster_writer.write(hUx, "hUx", t_out);
      raster_writer.write(hUy, "hUy", t_out);
      raster_writer.write(C, "C", t_out);
      t_out += dt_out;
    }

    if (time_controller.current() >= backup_time - t_small){
      std::cout << "Writing backup files" << std::endl;
      cuBackupWriter(h, "h_backup_", backup_time);
      cuBackupWriter(z, "z_backup_", backup_time);
      cuBackupWriter(hU, "hU_backup_", backup_time);
      cuBackupWriter(C, "C_backup_", backup_time);
      backup_time += backup_interval;
    }



  } while (!time_controller.is_end());

  std::cout << "Total runtime " << total_runtime << "ms" << std::endl;

  return 0;

}

