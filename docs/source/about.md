## About


### About Pypims

Pypims is built upon the open-source software [hipims-cuda](https://github.com/HEMLab/hipims). Pypims provides an user friendly Python-based interface
for users to prepare the inputs, run the hipims-cuda model and visualise the outputs.

### About HiPIMS-CUDA

HiPIMS standards for High-Performance Integrated hydrodynamic Modelling System, which is developed by members of [HEMLab](https://www.hemlab.org). There are three versions of hipims. hipims-cuda is one of them.
hipims-cuda uses state-of-art numerical schemes (Godunov-type finite volume) to solve the 2D shallow water equations for flood simulations. 
To support high-resolution flood simulations, hipims-cuda is implemented on multiple GPUs (Graphics Processing Unit) using CUDA/C++ languages to achieve high-performance computing. 
Since hipims-cuda has a modular and flexible structure, it has a great potential to be further 
developed for other applications in hydrological science as long as the problem can be solved on a uniform rectangular grid.