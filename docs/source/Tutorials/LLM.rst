==============================================
Using LLMs to Build SynxFlow Workflow
==============================================

This guide explains how you can leverage a Large Language Model (LLM) as your assistant when using SynxFlow. In this example, OpenAI's ChatGPT o3-mini is used because it is effective at coding tasks and can access internet information. (Other LLMs with similar capabilities can also be used.)

------------------------------------------------------------
Step 1: Browsing the Documentation (The First Prompt)
------------------------------------------------------------

This step ensures that the LLM learns the latest API usage and best practices of SynxFlow by reviewing its online documentation.

**First Prompt:**

.. admonition:: Example Prompt
   :class: tip

   “Please use your browsing capability to visit and read the following SynxFlow documentation pages and tutorials:

   1. InputModel API Documentation: `https://synxflow.readthedocs.io/en/latest/Modules/InputModel.html <https://synxflow.readthedocs.io/en/latest/Modules/InputModel.html>`_
   2. Boundary API Documentation: `https://synxflow.readthedocs.io/en/latest/Modules/Boundary.html <https://synxflow.readthedocs.io/en/latest/Modules/Boundary.html>`_
   3. Rainfall API Documentation: `https://synxflow.readthedocs.io/en/latest/Modules/Rainfall.html <https://synxflow.readthedocs.io/en/latest/Modules/Rainfall.html>`_
   4. OutputModel API Documentation: `https://synxflow.readthedocs.io/en/latest/Modules/OutputModel.html <https://synxflow.readthedocs.io/en/latest/Modules/OutputModel.html>`_
   5. Raster API Documentation: `https://synxflow.readthedocs.io/en/latest/Modules/Raster.html <https://synxflow.readthedocs.io/en/latest/Modules/Raster.html>`_
   6. Tutorials (practical examples of using SynxFlow): `https://synxflow.readthedocs.io/en/latest/Tutorials/flood.html <https://synxflow.readthedocs.io/en/latest/Tutorials/flood.html>`_
   7. Installation (steps to install SynxFlow): `https://synxflow.readthedocs.io/en/latest/quickstart.html <https://synxflow.readthedocs.io/en/latest/quickstart.html>`_

   After reviewing, summarise the key functionalities of SynxFlow.”

By reviewing these pages, the LLM gathers detailed information about:

- **InputModel:** How to define inputs (e.g., DEM, rainfall, boundaries).
- **Boundary:** How to set up different boundary conditions.
- **Rainfall:** How to handle time series and spatial rainfall data.
- **OutputModel:** How to read and analyze simulation outputs.
- **Raster:** How to process gridded data and visualize them.
- **Tutorials:** Step-by-step examples for running flood simulations.
- **Installtation:** Steps to install SynxFlow.

------------------------------------------------------------
Step 2: Setting Up a Simulation Case with LLM Assistance
------------------------------------------------------------

Once the documentation has been reviewed, you can ask the LLM to generate a sample code example for setting up a simulation. For example:

.. admonition:: Example Prompt
   :class: tip

   “Please show me a sample code that creates, runs, and visualizes a flood simulation using SynxFlow’s sample data.”

The LLM will generate a script that:
- Loads sample data (DEM, rain mask, landcover, rain source)
- Initializes an **InputModel** with the proper initial conditions, boundary conditions, and rainfall inputs
- Sets up landcover-based parameters (e.g., Manning’s n)
- Writes input files and runs the simulation using **flood.run()**
- Reads simulation outputs with **OutputModel** and visualizes them (using **Raster.mapshow()**)

------------------------------------------------------------
Step 3: Advanced Usage – Parameter Sensitivity Analysis
------------------------------------------------------------

To extend the simulation, you can ask the LLM to generate code for performing a sensitivity analysis. For instance, you might ask:

.. admonition:: Example Prompt
   :class: tip

   “Write code that loops over Manning’s n values from 0.01 to 0.10 (in increments of 0.01) that is uniform across the domain and plots the maximum inundation depth versus Manning’s n.”

The generated code will:
- Loop over the range of Manning’s n values.
- For each value, create a unique simulation case folder and run a simulation.
- Post-process the output to extract a key metric (e.g., maximum water depth).
- Plot the sensitivity of the maximum depth with respect to Manning’s n using Matplotlib.

When I tested it, the generated code was

.. code-block:: python

   import os
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt

   # Import SynxFlow modules (ensure synxflow is installed and accessible)
   from synxflow import IO, flood

   # ------------------------------------------------------------------------------
   # 1. Load Sample Data (common to all runs)
   # ------------------------------------------------------------------------------
   # Retrieve sample data (DEM, rain mask, landcover, and rain source)
   dem_file, demo_data, data_path = IO.demo_functions.get_sample_data()
   DEM = IO.Raster(os.path.join(data_path, 'DEM.gz'))
   rain_mask = IO.Raster(os.path.join(data_path, 'rain_mask.gz'))
   landcover = IO.Raster(os.path.join(data_path, 'landcover.gz'))
   # Read the rainfall time series (CSV) and convert to a NumPy array
   rain_source = pd.read_csv(os.path.join(data_path, 'rain_source.csv'), header=None).to_numpy()

   # Define common boundary conditions and gauge positions
   box_upstream   = np.array([[1427, 195], [1446, 243]])
   box_downstream = np.array([[58, 1645], [72, 1170]])
   discharge_values = np.array([[0, 100],
                              [3600, 100]])
   boundary_list = [
      {'polyPoints': box_upstream, 'type': 'open', 'hU': discharge_values},
      {'polyPoints': box_downstream, 'type': 'open', 'h': np.array([[0, 5], [3600, 5]])}
   ]
   gauges = np.array([[560, 1030],
                     [1140, 330]])

   # Define simulation runtime parameters:
   runtime = [0, 7200, 900, 1800]  # [start time, end time, output interval, backup interval]

   # ------------------------------------------------------------------------------
   # 2. Loop over Uniform Manning's n Values and Run Simulations
   # ------------------------------------------------------------------------------
   mannings = np.arange(0.02, 0.11, 0.01)  # Manning's n values: 0.01, 0.02, ..., 0.10
   max_depths = []  # to store the maximum inundation depth for each run

   for m in mannings:
      print(f"Running simulation for uniform Manning's n = {m:.2f}")

      # Create a unique folder for each simulation run
      case_folder = os.path.join(os.getcwd(), f'flood_case_m{m:.2f}')
      if not os.path.exists(case_folder):
         os.makedirs(case_folder)

      # ------------------------------------------------------------------------------
      # Set up the simulation inputs using the InputModel
      # ------------------------------------------------------------------------------
      # Create the InputModel using the DEM as the reference grid.
      # Here, num_of_sections=1 indicates a single-GPU (or single-domain) simulation.
      case_input = IO.InputModel(DEM, num_of_sections=1, case_folder=case_folder)

      # Set the initial condition: water depth h0 is 0 everywhere.
      case_input.set_initial_condition('h0', 0.0)

      # Set the boundary conditions.
      case_input.set_boundary_condition(boundary_list=boundary_list)

      # Set the rainfall inputs using the rain mask and the rain source time series.
      case_input.set_rainfall(rain_mask=rain_mask, rain_source=rain_source)

      # Set the landcover data.
      case_input.set_landcover(landcover)

      # Set a uniform Manning's n value over the entire domain.
      # For a uniform value, you can simply pass the scalar.
      case_input.set_grid_parameter(manning=m)

      # Set gauge positions.
      case_input.set_gauges_position(gauges)

      # Set simulation runtime parameters.
      case_input.set_runtime(runtime)

      # (Optional) Visualize the domain and boundaries for verification.
      # case_input.domain_show()

      # Write all the necessary input files for the simulation.
      case_input.write_input_files()

      # ------------------------------------------------------------------------------
      # Run the flood simulation
      # ------------------------------------------------------------------------------
      # Use flood.run() for a single-GPU run.
      flood.run(case_folder)

      # ------------------------------------------------------------------------------
      # Post-Processing: Extract the Maximum Inundation Depth
      # ------------------------------------------------------------------------------
      # Create an OutputModel object to read simulation outputs.
      case_output = IO.OutputModel(input_obj=case_input)

      # Read the grid output (for example, assume 'h_max_7200' holds the maximum water depth at t = 7200 s).
      max_depth_raster = case_output.read_grid_file(file_tag='h_max_7200')

      # Extract the maximum depth value from the raster array.
      max_val = np.nanmax(max_depth_raster.array)
      max_depths.append(max_val)

      print(f"Uniform Manning's n = {m:.2f} results in a maximum inundation depth of {max_val:.2f} m")

   # ------------------------------------------------------------------------------
   # 3. Plot Maximum Inundation Depth vs. Uniform Manning's n
   # ------------------------------------------------------------------------------
   plt.figure(figsize=(8, 5))
   plt.plot(mannings, max_depths, marker='o', linestyle='-')
   plt.xlabel("Uniform Manning's n")
   plt.ylabel("Maximum Inundation Depth (m)")
   plt.title("Sensitivity of Maximum Inundation Depth to Uniform Manning's n")
   plt.grid(True)
   plt.show()

------------------------------------------------------------
Step 4: Visualizing Gridded Outputs with Built-In Functions
------------------------------------------------------------

ChatGPT can instruct you on using SynxFlow’s built-in visualization functions. For example, to visualize a raster output such as the maximum inundation depth from your simulation, you might ask:

.. admonition:: Example Prompt
   :class: tip

   “How do I use the built-in functions to visualize gridded outputs in SynxFlow?”

ChatGPT may explain that you can use:
- **Raster.mapshow()** for a quick display of raster outputs.
- **Raster.hillshade()** for enhanced visualization of topography.

A sample code snippet might be:

.. code-block:: python

   # Read a gridded output (e.g., maximum water depth) and display it
   max_depth = case_output.read_grid_file(file_tag='h_max_7200')
   max_depth.mapshow(title='Maximum Inundation Depth')

------------------------------------------------------------
Step 5: Best Practices for Using LLMs with SynxFlow
------------------------------------------------------------

- **Choose the most suitable LLMs:**  
  Select LLMs that can reason and browse the internet (e.g., ChatGPT o3-mini, DeepSeek R1).

- **Ask LLMs to learn as the first step:**  
  Always instruct the LLM to gather the latest information from the online documentation first.

- **Be Specific:**  
  Clearly state your requirements (e.g., “Loop over Manning’s n values…” or “Explain fall boundaries…”).

- **Iterative Refinement:**  
  Use follow-up questions to get more detailed explanations or adjustments.

- **Combine Resources:**  
  Use the LLM in conjunction with the official SynxFlow documentation for a comprehensive understanding.

- **Customize Examples:**  
  Use the generated code as a starting point and adjust it to fit your project requirements.

------------------------------------------------------------
Conclusion
------------------------------------------------------------

By following these steps, you can efficiently harness the power of LLMs to build, refine, and analyze SynxFlow workflows. LLMs can help generate code, extend simulations with advanced analyses, and even produce visualization scripts, all while being informed by the latest official documentation.

