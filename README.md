# Python APIs for hipims

This package provides python APIs for running the open source hydraulic model [hipims-cuda](https://github.com/HEMLab/hipims). It also includes the [hipims-io](https://pypi.org/project/hipims-io/) package for pre-processing and result visualisation. The full documentation is at [https://pypims.readthedocs.io/en/latest/](https://pypims.readthedocs.io/en/latest/).

## Installation

Before installing pypims, you need to install [Cmake](https://cmake.org/download/), NVIDIA CUDA Toolkit and the C++ compiler. Cmake can be simply installed by

```shell
pip install cmake
```

The CUDA Toolkit can be downloaded from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads). If you use Linux system, the C++ compiler 'GCC' is usually on the system as default. If you use windows, you should download and install [Visual Studio](https://visualstudio.microsoft.com/vs/). If there exists several different versions of CUDA Toolkits on your system, cmake may struggle to decide which one to use and return an error. In such a case, you need to define the CUDAToolkit_ROOT environmental varialble. On Linux, this can be done by

```shell
export CUDAToolkit_ROOT=<path to your cuda directory>
```
For example

```shell
export CUDAToolkit_ROOT=/usr/local/cuda-10.1
```

Once the aforementioned dependencies have been properly installed. Installing pypims is straightforward, simple type in the following in your terminal

```shell
pip install pypims
```



## Basic usage

The flood simulation engine can be imported by

```python
from pypims import flood
```

The inputs can be prepared by [hipims-io](https://pypi.org/project/hipims-io/). This package has already been included in pypims, you can import it by

```python
from pypims import IO
```
A quick demonstration to prepare input files with attached sample data contaning the following files:
- DEM.gz/.asc/.tif (essential file, in projected crs, map unit:m)
- rain_mask.gz/.asc/.tif (optional file for setting rainfall, having the same crs with DEM)
- rain_source.csv (optional file for setting rainfall rate in timeseries, unit:m/s]
- landcover.gz/.asc/.tif (optional file for setting landcover-based parameters, having the same crs with DEM)

```python
import os
from pypims.IO.demo_functions import get_sample_data
data_path = get_sample_data(return_path=True) # get the path of sample data
case_folder = os.path.join(os.getcwd(), 'model_case') # define a case folder in the current directory
num_of_devices = 1
obj_in = IO.InputHipims(case_folder=case_folder, num_of_sections=num_of_devices, 
                        data_path=data_path) # create input object
obj_in.domain_show() # show domain map
print(obj_in) # show case information
obj_in.write_input_files() # create all input files
```

Once the inputs have all been prepared, the simulations can be started by


```python
flood.run(case_folder)
```

or

```python
flood.run_mgpus(case_folder) # using this if num_of_devices > 1
```

for multiple GPUs.