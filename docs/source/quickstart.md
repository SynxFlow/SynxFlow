## Getting started

### Installing

You will have to set up your Python environment as a first step. There are several ways of doing it, but [Anaconda](https://www.anaconda.com/products/distribution) is a highly recommended way. 

Before installing pypims, you also need to install [Cmake](https://cmake.org/download/), NVIDIA CUDA Toolkit and the C++ compiler. Cmake can be simply installed by

```shell
pip install cmake
```

Pypims needs NVIDIA GPUs which support CUDA to run the simulations, therefore you need the GPU and the CUDA Toolkit. The CUDA Toolkit can be downloaded from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads). If you use Linux system, the C++ compiler 'GCC' is usually on the system as default. If you use windows, you should download and install [Visual Studio](https://visualstudio.microsoft.com/vs/).

#### Installing on Linux

You may need to define the CUDAToolkit_ROOT environmental varialble first if there exists several different versions of CUDA Toolkits on your system. This can be done by

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

#### Installing on Windows

Installing on Windows is slightly less straightforward than on Linux. The steps below are tested with Windows 10, Visual Studio 2019 and Python 3.7. You need to firstly manually install the dependency packages inculding 'GDAL', 'rasterio' and 'fiona'. 

Taking GDAL as an example, firstly download the wheel from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/). You need to choose the wheel for the right platform and python version, for example 'GDAL‑3.4.2‑cp37‑cp37m‑win_amd64.whl', which corresponds to amd64 platform and Python 3.7. Then GDAL can be installed in the commandline tool as

```shell
python -m pip install <path to downloaded wheel>
```

'rasterio' and 'fiona' need to be installed following similar steps.

The next step is to download the source code of pypims from [Github](https://github.com/pypims/pypims). This can also be done by

```shell
git clone https://github.com/pypims/pypims.git
```

After finishing downloading, go to the folder of the pypims source code and run the following command

```shell
python setup.py install
```
Now you should have pypims successfully installed on your computer.

### Running a simulation with the example

The flood simulation engine can be imported by

```python
from pypims import flood
```

The inputs can be prepared by [hipims-io](https://pypi.org/project/hipims-io/). This package has already been included as part of pypims, you can import it by

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
from pypims.IO.demo_functions import demo_input
obj_in = demo_input() # create input object
obj_in.write_input_files() # create all input files
```

Once the inputs have all been prepared, the simulations can be started by


```python
flood.run(obj_in.get_case_folder())
```
