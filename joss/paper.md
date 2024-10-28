---
title: 'SynxFlow: A GPU-accelerated Python package for multi-hazard simulations'
tags:
  - Python
  - flood
  - landslide
  - debris flow
  - GPU
authors:
  - name: Xilin Xia
    orcid: 0000-0002-5784-9211
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Xiaodong Ming
    orcid: 0000-0001-9114-2642
    affiliation: 2
affiliations:
 - name: School of Engineering, University of Birmingham, Birmingham, UK
   index: 1
 - name: Ping An P&C Insurance Company of China, Shenzhen, China
   index: 2

date: 27 October 2024
bibliography: paper.bib
---

# Summary

SynxFlow can dynamically simulate flood inundation, landslides runout and debris flows using multiple CUDA-enabled GPUs. It also offers a Python interface for users to set up simulations and visualise outputs. The software is interoperable with popular packages such as numpy, pandas and rasterio. Therefore, SynxFlow can be integrated into data science workflows to streamline and accelerate hazard risk assessment tasks.

# Statement of need

Natural hazards like floods, landslides, and debris flows cause significant damage each year, and due to climate change, the frequency and severity of these events are increasing. This rising threat highlights an urgent need to develop tools that can enhance our understanding of these risks, supporting crucial applications like early warning systems and comprehensive risk assessments.

Effectively assessing these risks requires integrating data from multiple sources—such as climate, geological, hydrological, and environmental data, while also accounting for complex, interacting hazards. For comprehensive risk assessment over a large region, processing vast amounts of data swiftly is essential, often necessitating cloud-based computing or the use of high-performance computing (HPC) systems. These HPC systems, especially when enhanced with Graphics Processing Units (GPUs), allow faster processing of complex simulations. However, most existing software for simulating natural hazards is desktop-based and challenging to integrate into the modern data science ecosystem, making it unsuitable for large-scale, real-time applications and collaborative research efforts. This gap underscores the need for advanced and scalable tools that leverage modern computing infrastructure, enabling seamless integration, real-time data processing, and accessibility for diverse stakeholders.

To this end, we believe modern multi-hazard simulation software should be

1. Performant – The software should utilise modern GPU-based HPC systems to accelerate complex simulations, enabling faster data processing and real-time hazard prediction. This is crucial for rapid risk assessment over large regions with multiple hazards.
2. Accessible – The software should be open-source and easy-to-deploy, allowing transparency and collaboration. Accessible deployment options also lower infrastructure barriers, enabling use in diverse settings.
3. Interoperable – The software should integrate seamlessly with popular data science tools to leverage advanced analytics and machine learning. It should also enable easy coupling of various hazard simulations. This is crucial for comprehensive and effective risk analytics.

The SynxFlow project is our attempt to achieve this vision.

# Design and Implementation

The software is a Python package with the following two parts:

## Solvers
There are three modules, flood, landslide and debris, solving the shallow water type equations using CUDA-enabled multiple GPUs, to simulate flood inundation, landslides runout and debris flows respectively. The solvers can predict the depth and velocities of these hazards. The flood solver, inherited from Pypims[@Xia2023b], is a Python-bound version of the open-source code hipims-cuda[@HEMLab2020]. Since the start of the SynxFlow project, the landslide solver of 'hipims-cuda' has also been Python-bound and added to SynxFlow, and a debris flow solver has been implemented. These solvers use Godunov-type shock-capturing finite volume methods, which are documented in [@Xia2017; @Xia2018d; @Xia2023]. The numerical methods have been extensively tested in both theoretical and real-world scenarios, demonstrating robustness and efficiency for practical applications.

## IO (Input and Output)
The IO module comprises of several classes and functions that generates the inputs to the solvers and process their outputs. The IO functionality for the flood solver is based on the open-source 'hipims-io'[@Ming2023] package, which has been extended to support the landslide and debris solvers. To set up a simulation, the users need to define an object using the 'InputModel' class, using common data formats including numpy arrays and pandas dataframes among others. An 'OutputModel' type object can be used to perform a variety of visualisation tasks including plotting maps, hydrographs and creating animations. To assist users, the IO module also provide functionalities to process geospatial data such as rasters and shapefiles. The IO module effectively integrates the solvers into data science workflows, streamlining hazard risk assessment tasks, and also enables coupling between solvers to facilitate multi-hazard simulations.

Compared with existing software, SynxFlow is unique for its combination of open-source accessibility, multi-GPU acceleration, multi-hazard simulation capabilities, and a Python-based user interface.

# Acknowledgements

The development of the code was enabled by the Baskerville Tier-2 HPC, which is funded by the EPSRC and UKRI through the World Class Labs scheme (EP/T022221/1) and the Digital Research Infrastructure programme (EP/W032244/1).

# References