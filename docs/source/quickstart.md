## Getting started

### Installing

#### First steps

You will have to set up your Python environment as a first step. There are several ways of doing it, but [Anaconda](https://www.anaconda.com/products/distribution) is a highly recommended way. 


synxflow also needs NVIDIA GPUs which support CUDA to run the simulations, therefore you need a GPU and the CUDA Toolkit. The CUDA Toolkit can be downloaded from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads). Please follow their websites about installation of CUDA Toolkit. 

#### Installing on Linux

Installing on Linux is done by compiling the package from source code. Before installing synxflow, you also need to install [Cmake](https://cmake.org/download/), NVIDIA CUDA Toolkit and the C++ compiler. The C++ compiler 'GCC' is usually on the system as default. Cmake can be simply installed by

```shell
pip install cmake
```



You may need to define the CUDAToolkit_ROOT environmental varialble first if there exists several different versions of CUDA Toolkits on your system. This can be done by

```shell
export CUDAToolkit_ROOT=<path to your cuda directory>
```
For example

```shell
export CUDAToolkit_ROOT=/usr/local/cuda-11.3
```

Once the aforementioned dependencies have been properly installed. Installing synxflow is straightforward, simple type in the following in your terminal

```shell
pip install synxflow
```

#### Installing on Windows

Installing on Windows is also strightforward because it uses pre-built wheel. To successfully install synxflow, you would need Python 3.9 or 3.10 and CUDA 11.3 or newer. Then synxflow can be installed by

```shell
pip install synxflow
```

Now you should have synxflow successfully installed on your computer.

### Running a simulation with the example

The flood simulation engine can be imported by

```python
from synxflow import flood
```

The IO package for processing inputs and outputs can be imported by

```python
from synxflow import IO
```
A quick demonstration to prepare input files with attached sample data contaning the following files:
- DEM.gz/.asc/.tif (essential file, in projected crs, map unit:m)
- rain_mask.gz/.asc/.tif (optional file for setting rainfall, having the same crs with DEM)
- rain_source.csv (optional file for setting rainfall rate in timeseries, unit:m/s]
- landcover.gz/.asc/.tif (optional file for setting landcover-based parameters, having the same crs with DEM)

```python
import os
from synxflow.IO.demo_functions import demo_input
obj_in = demo_input() # create input object
obj_in.write_input_files() # create all input files
```

Once the inputs have all been prepared, the simulations can be started by


```python
flood.run(obj_in.get_case_folder())
```
