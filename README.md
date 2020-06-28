# PYPIMS

This package provides python APIs for running the open source hydraulic model [hipims-cuda](https://github.com/HEMLab/hipims). It also includes the [hipims-io](https://pypi.org/project/hipims-io/) package for pre-processing and result visualisation.

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

Once the inputs have all been prepared, the simulations  can be started by

```python
flood.run('path to your inputs')
```

or

```python
flood.run_mgpus('path to your inputs')
```

for multiple GPUs.