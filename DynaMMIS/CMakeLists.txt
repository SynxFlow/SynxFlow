find_package(CUDA)

IF(WIN32)
    set(PLATFORM_CXX_FLAGS /Zi)
    set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_NVCC_FLAGS -arch=sm_35;--expt-extended-lambda)
ENDIF(WIN32)

IF(UNIX)
    set(PLATFORM_CXX_FLAGS "-pthread -std=c++11 -fPIC -O3")
    #set(CUDA_SEPARABLE_COMPILATION ON)
    set(CUDA_NVCC_FLAGS  -arch=sm_35;--expt-extended-lambda)
ENDIF(UNIX)

list(APPEND CMAKE_CXX_FLAGS ${PLATFORM_CXX_FLAGS})

message("CXX  Flags: " ${CMAKE_CXX_FLAGS})
message("NVCC Flags: " ${CUDA_NVCC_FLAGS})

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)

include(FetchContent)

FetchContent_Declare(
  pybind11_sources
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.5
)

FetchContent_GetProperties(pybind11_sources)

if(NOT pybind11_sources_POPULATED)
  FetchContent_Populate(pybind11_sources)

  add_subdirectory(
    ${pybind11_sources_SOURCE_DIR}
    ${pybind11_sources_BINARY_DIR}
    )
endif()

message("pybind11 : "${pybind11_sources_SOURCE_DIR})

add_subdirectory(lib)
add_subdirectory(apps/cudaFloodSolversPybind)
#add_subdirectory(apps/cudaMultiGPUsFloodSolver)
#add_subdirectory(apps/cudaIntegratedShallowFlowSolver)
