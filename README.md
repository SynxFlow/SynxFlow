# HiPIMS-PYTHON

This package provides python APIs for running the open source hydraulic model [hipims-cuda](https://github.com/HEMLab/hipims). It is used in companion with the [hipims-io](https://pypi.org/project/hipims-io/) package for pre-processing and result visualisation.

## Installation

Before installing hipims, you need to install [Cmake](https://cmake.org/download/), NVIDIA CUDA Toolkit and the C++ compiler. Cmake can be simply installed by

```shell
pip install cmake
```

The CUDA Toolkit can be downloaded from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads). If you use Linux system, the C++ compilerGCC is usually on the system as default. If you use windows, you should download and install [Visual Studio](https://visualstudio.microsoft.com/vs/).

Once the aforementioned dependencies have been properly installed. Installing hipims is straightforward, simple type in the following in your terminal

```shell
pip install hipims
```



## Basic usage

The package can be imported by

```python
from hipims import flood
```

The inputs to hipims can be prepared by [hipims-io](https://pypi.org/project/hipims-io/). Once the inputs have all been prepared, the simulations  can be started by

```python
flood.run('path to your inputs')
```

or

```python
flood.run_mgpus('path to your inputs')
```

for multiple GPUs.