# Using LLMs to Build SynxFlow Workflow

Authour: [Xilin Xia](https://www.linkedin.com/in/xilin-xia-8787a4199/?originalSubdomain=uk)

This guide explains how you can leverage Large Large Language Model as your assistant for using SynxFlow. I will use OpenAI's ChatGPT o3-mini as an example, which is good at coding tasks and can access information on the internet. But you can also use other LLMs with similar capabilities.

---

## Step 1: Browsing the Documentation (The First Prompt)

This is to ensure that ChatGPT can learn the latest information about the API usage of SynxFlow.

**First Prompt:**

> *“Please use your browsing capability to visit and read the following SynxFlow documentation pages and tutorials:*

> *1. InputModel API Documentation: [https://synxflow.readthedocs.io/en/latest/Modules/InputModel.html](https://synxflow.readthedocs.io/en/latest/Modules/InputModel.html)*  
> *2. Boundary API Documentation: [https://synxflow.readthedocs.io/en/latest/Modules/Boundary.html](https://synxflow.readthedocs.io/en/latest/Modules/Boundary.html)*  
> *3. Rainfall API Documentation: [https://synxflow.readthedocs.io/en/latest/Modules/Rainfall.html](https://synxflow.readthedocs.io/en/latest/Modules/Rainfall.html)*  
> *4. OutputModel API Documentation: [https://synxflow.readthedocs.io/en/latest/Modules/OutputModel.html](https://synxflow.readthedocs.io/en/latest/Modules/OutputModel.html)*  
> *5. Raster API Documentation: [https://synxflow.readthedocs.io/en/latest/Modules/Raster.html](https://synxflow.readthedocs.io/en/latest/Modules/Raster.html)*  
> *6. Tutorials: (for practical examples of using SynxFlow) [https://synxflow.readthedocs.io/en/latest/Tutorials/flood.html](https://synxflow.readthedocs.io/en/latest/Tutorials/flood.html)*  
> *7. Installation: (for steps of install the model) [https://synxflow.readthedocs.io/en/latest/quickstart.html](https://synxflow.readthedocs.io/en/latest/quickstart.html)*  

> *After reviewing, summarise key functionalities of SynxFlow.”*



This first prompt instructs ChatGPT to browse and understand the key components of SynxFlow. By reviewing these pages, ChatGPT gathers detailed information about:

- **InputModel:** How to define inputs (DEM, rainfall, boundaries, etc.)  
- **Boundary:** How to set up different boundary conditions (including the "fall" type)  
- **Rainfall:** How to handle time series and spatial distribution of rainfall  
- **OutputModel:** How to read and analyze simulation outputs  
- **Raster:** How to handle gridded data and visualize them
- **Tutorials:** Practical, step-by-step examples for running flood simulations

This step is crucial because it ensures that any examples or code generated are informed by the latest SynxFlow documentation and best practices.

---

## Step 2: Setting Up a Simulation Case with ChatGPT Assistance

Once the documentation has been reviewed, you can ask ChatGPT to generate a code example for setting up a simulation. For instance, you might ask:

> *“Please show me a sample code that creates, runs, and visualizes a flood simulation using SynxFlow’s sample data.”*


ChatGPT will then provide a complete script that:
- Loads sample data (DEM, rain mask, landcover, rain source)  
- Initializes an **InputModel** with proper initial conditions, boundary conditions, and rainfall inputs  
- Sets up landcover-based parameters (such as Manning’s n)  
- Writes input files and runs the simulation using **flood.run()**  
- Reads simulation outputs with **OutputModel** and visualizes them (for example, using **Raster.mapshow()**)

---

## Step 3: Advanced Usage – Parameter Sensitivity Analysis

ChatGPT can further help you extend your simulation to perform sensitivity analyses. For example, if you wish to study the impact of varying Manning’s n from 0.01 to 0.1, you might ask:


> *“Write code that loops over Manning’s n values from 0.01 to 0.10 (in increments of 0.01) and plots the maximum inundation depth versus Manning’s n.”*


The provided code will:
- Loop over the range of Manning’s n values  
- For each value, create a unique simulation case folder and run a simulation  
- Post-process the output to extract a metric (e.g., the overall maximum water depth)  
- Plot the sensitivity of the maximum depth with respect to Manning’s n using Matplotlib

Below is an example code snipped when I tested it:
```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import SynxFlow modules (assumes synxflow is installed and in PYTHONPATH)
from synxflow import IO, flood

# ------------------------------------------------------------------------------
# 1. Load Sample Data (same for all runs)
# ------------------------------------------------------------------------------
# Get sample data (DEM, rain_mask, landcover, and rain_source)
dem_file, demo_data, data_path = IO.demo_functions.get_sample_data()
DEM = IO.Raster(os.path.join(data_path, 'DEM.gz'))
rain_mask = IO.Raster(os.path.join(data_path, 'rain_mask.gz'))
landcover = IO.Raster(os.path.join(data_path, 'landcover.gz'))
# Read the rainfall time series from CSV and convert to NumPy array
rain_source = pd.read_csv(os.path.join(data_path, 'rain_source.csv'), header=None).to_numpy()

# Define common boundary conditions and gauge positions (as in the tutorial)
box_upstream = np.array([[1427, 195], [1446, 243]])
box_downstream = np.array([[58, 1645], [72, 1170]])
discharge_values = np.array([[0, 100],
                             [3600, 100]])
bound_list = [
    {'polyPoints': box_upstream,
     'type': 'open',
     'hU': discharge_values},
    {'polyPoints': box_downstream,
     'type': 'open',
     'h': np.array([[0, 5],
                    [3600, 5]])}
]
gauges = np.array([[560, 1030],
                   [1140, 330]])

# Simulation runtime: [start, end, output_interval, backup_interval]
runtime = [0, 7200, 900, 1800]

# ------------------------------------------------------------------------------
# 2. Loop over Manning's n values and run simulations
# ------------------------------------------------------------------------------
mannings = np.arange(0.01, 0.11, 0.01)
max_depths = []  # to store maximum inundation depths for each run

# Loop over each Manning's n value
for m in mannings:
    print(f"Running simulation for Manning's n = {m:.2f}")
    
    # Create a unique case folder for this simulation run
    case_folder = os.path.join(os.getcwd(), f'flood_case_m{m:.2f}')
    if not os.path.exists(case_folder):
        os.makedirs(case_folder)
    
    # ------------------------------------------------------------------------------
    # Set up the simulation inputs
    # ------------------------------------------------------------------------------
    # Create an InputModel object using the DEM; use one GPU (num_of_sections=1)
    case_input = IO.InputModel(DEM, num_of_sections=1, case_folder=case_folder)
    
    # Set initial condition: water depth (h0) is 0 everywhere.
    case_input.set_initial_condition('h0', 0.0)
    
    # Set boundary conditions
    case_input.set_boundary_condition(boundary_list=bound_list)
    
    # Set rainfall inputs using the rain mask and rain source data
    case_input.set_rainfall(rain_mask=rain_mask, rain_source=rain_source)
    
    # Set the landcover data
    case_input.set_landcover(landcover)
    
    # Set the grid parameter for Manning's n uniformly across the domain.
    # Here, both param_value and default_value are set to the current m.
    case_input.set_grid_parameter(manning={'param_value': m, 'default_value': m})
    
    # Set gauge positions
    case_input.set_gauges_position(gauges)
    
    # Set simulation runtime parameters
    case_input.set_runtime(runtime)
    
    # (Optional) Display the domain with boundary overlays for verification
    # case_input.domain_show()
    
    # Write all input files required by the simulation engine.
    case_input.write_input_files()
    
    # ------------------------------------------------------------------------------
    # Run the simulation
    # ------------------------------------------------------------------------------
    # For a single GPU, use flood.run.
    flood.run(case_folder)
    
    # ------------------------------------------------------------------------------
    # Post-process: read the simulation output and extract maximum inundation depth
    # ------------------------------------------------------------------------------
    # Create an OutputModel object to read outputs
    case_output = IO.OutputModel(input_obj=case_input)
    
    # Read the grid output that contains the maximum water depth.
    # (Assuming the file tag 'h_max_7200' corresponds to water depth at the final time.)
    max_depth_raster = case_output.read_grid_file(file_tag='h_max_7200')
    
    # Get the maximum depth from the output raster array.
    # It is assumed that 'array' is the NumPy array of water depths.
    max_val = np.nanmax(max_depth_raster.array)
    max_depths.append(max_val)
    
    print(f"Manning's n = {m:.2f} --> Maximum Inundation Depth = {max_val:.2f} m")

# ------------------------------------------------------------------------------
# 3. Plot Maximum Inundation Depth vs. Manning's n
# ------------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(mannings, max_depths, marker='o', linestyle='-')
plt.xlabel("Manning's n")
plt.ylabel("Maximum Inundation Depth (m)")
plt.title("Sensitivity of Maximum Inundation Depth to Manning's n")
plt.grid(True)
plt.show()
```

---

## Step 4: Visualizing Gridded Outputs with Built-In Functions

ChatGPT can also instruct you on using SynxFlow’s built-in visualization functions. For example, if you want to visualize a raster output such as the maximum inundation depth from your simulation, you might ask:


> *“How do I use the built-in functions to visualize gridded outputs in SynxFlow?”*


ChatGPT may explain that you can use:
- **`Raster.mapshow()`** for a quick display of raster outputs  
- **`Raster.hillshade()`** for enhanced visualizations of topography

A sample snippet might be:

```python
# Read a gridded output (e.g., maximum water depth) and display it
max_depth = case_output.read_grid_file(file_tag='h_max_7200')
max_depth.mapshow(title='Maximum Inundation Depth')
```

---

## Step 5: Best Practices for Using LLMs with SynxFlow

- **Choose the most suitable LLMs:** Chose those that can reason and browse internet such as ChatGPT o3-mini, or DeepSeek R1.
- **Ask LLMs to learn as the first step:** Always ask LLMs to gather the latest information from the online document as the first step.
- **Be Specific:** Clearly state what you need (e.g., “Loop over Manning’s n values…” or “Explain fall boundaries…”).
- **Iterative Refinement:** Use follow-up questions to get more detailed explanations or adjustments.  
- **Combine Resources:** Use ChatGPT alongside the official SynxFlow documentation to get a comprehensive understanding.  
- **Customize Examples:** Use the generated code as a starting point and adjust it to fit your project requirements.


---

## Conclusion

By following these steps, you can efficiently harness LLMs capabilities to work with SynxFlow:

1. **Start by Browsing the Documentation:**  
   Begin with the prompt that instructs ChatGPT to read the key SynxFlow pages—including the InputModel API at [https://synxflow.readthedocs.io/en/latest/Modules/InputModel.html](https://synxflow.readthedocs.io/en/latest/Modules/InputModel.html) and others.
2. **Generate and Customize Code Examples:**  
   Get complete scripts for simulations, visualizations, and sensitivity analyses.
3. **Perform Advanced Analyses:**  
   Use ChatGPT to create loops, post-process outputs, and plot results.

Finally, LLMs can still make mistakes, but it can be a great assistant to accelerate your tasks if you use it wisely.