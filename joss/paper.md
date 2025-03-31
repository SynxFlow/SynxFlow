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

Effectively assessing these risks requires integrating data from multiple sources—such as climate, geological, hydrological, and environmental data, while also accounting for complex, interacting hazards. For comprehensive risk assessment over a large region, processing vast amounts of data swiftly is essential, often necessitating cloud-based computing or the use of high-performance computing (HPC) systems. These HPC systems, especially when enhanced with Graphics Processing Units (GPUs), allow faster processing of complex simulations.

However, current modeling tools show significant limitations in meeting the growing demands of multi-hazard risk assessment and in leveraging modern computing technologies. For example, tools like LISFLOOD-FP 8.0 [@Shaw2021] and TRITON [@MoralesHernandez2021] offer GPU acceleration, but their scope is restricted to flood simulations and they lack native Python interfaces — making integration with data science workflows cumbersome. GeoClaw [@clawpack;@BergerLeVeque98] supports multiple hazards and is Python-compatible, yet its single-GPU acceleration makes it inefficient for large-scale, high-resolution applications.

HEC-RAS [@HEC-RAS] is widely used due to its user-friendly graphical interface and support for third-party Python automation. However, it does not support multi-GPU acceleration, cannot simulate landslide runout, and remains proprietary software — limiting both customisation and open development. These constraints highlight a critical gap: existing tools are either too specialised, insufficiently scalable, or poorly integrated with modern data and compute ecosystems.

To this end, we believe modern multi-hazard simulation software should be

1. Performant – The software should utilise modern GPU-based HPC systems to accelerate complex simulations, enabling faster data processing and real-time hazard prediction. This is crucial for rapid risk assessment over large regions with multiple hazards.
2. Accessible – The software should be open-source and easy-to-deploy, allowing transparency and collaboration. Accessible deployment options also lower infrastructure barriers, enabling use in diverse settings.
3. Interoperable – The software should integrate seamlessly with popular data science tools to leverage advanced analytics and machine learning. It should also enable easy coupling of various hazard simulations. This is crucial for comprehensive and effective risk analytics.

The SynxFlow project is our attempt to achieve this vision.

# Design and Implementation

The software is a Python package with the following two parts:

## Solvers
There are three modules, flood, landslide and debris, solving the shallow water type equations using CUDA-enabled multiple GPUs, to simulate flood inundation, landslides runout and debris flows respectively. The solvers can predict the depth and velocities of these hazards. The flood solver, inherited from Pypims [@Xia2023b], is a Python-bound version of the open-source code hipims-cuda [@HEMLab2020]. Since the start of the SynxFlow project, the landslide solver of hipims-cuda has also been Python-bound and added to SynxFlow, and a debris flow solver has been implemented. These solvers use Godunov-type shock-capturing finite volume methods, which are documented in @Xia2017, @Xia2018d and @Xia2023. The numerical methods have been extensively tested in both theoretical and real-world scenarios, demonstrating robustness and efficiency for practical applications.

## IO (Input and Output)
The IO module comprises of several classes and functions that generates the inputs to the solvers and process their outputs. The IO functionality for the flood solver is based on the open-source hipims-io [@Ming2023] package, which has been extended to support the landslide and debris solvers. To set up a simulation, the users need to define an object using the 'InputModel' class, using common data formats including numpy arrays and pandas dataframes among others. An 'OutputModel' type object can be used to perform a variety of visualisation tasks including plotting maps, hydrographs and creating animations. To assist users, the IO module also provide functionalities to process geospatial data such as rasters and shapefiles. The IO module effectively integrates the solvers into data science workflows, streamlining hazard risk assessment tasks, and also enables coupling between solvers to facilitate multi-hazard simulations. The Python-based IO module also enables controlling scripts to be generated by Large Language Models (LLMs) such as ChatGPT and Gemini, which offers huge potential for automation.

Compared with existing software, SynxFlow is unique for its combination of open-source accessibility, multi-GPU acceleration, multi-hazard simulation capabilities, and a Python-based user interface.

# Acknowledgements

The development of the code was enabled by the Baskerville Tier-2 HPC, which is funded by the EPSRC and UKRI through the World Class Labs scheme (EP/T022221/1) and the Digital Research Infrastructure programme (EP/W032244/1).

# References