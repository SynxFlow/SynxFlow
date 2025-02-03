==========================================
SynxFlow Engines API Documentation
==========================================

The SynxFlow package includes several simulation engines implemented as C extensions,
each responsible for modeling different types of natural hazard scenarios. This document
provides an overview of the three main engines:

- **Flood Simulation Engine**
- **Landslide Runout Engine**
- **Debris Flow Simulation Engine**

Each engine module is designed to leverage GPU acceleration for high-performance simulations.
The following sections describe the purpose, key function(s), parameters, return values, and
usage examples for each engine.==========================================
SynxFlow Engines API Documentation
==========================================

The SynxFlow package includes several simulation engines implemented as C extensions,
each responsible for modeling different types of natural hazard scenarios. This document
provides an overview of the three main engines:

- **Flood Simulation Engine**
- **Landslide Runout Engine**
- **Debris Flow Simulation Engine**

Each engine module is designed to leverage GPU acceleration for high-performance simulations.
The following sections describe the purpose, key function(s), parameters, return values, and
usage examples for each engine.

flood
-----------------------
The **flood** module provides the core functionality for simulating flood inundation events.
It models water flow dynamics and flood extents using high-performance GPU computations.

**Key Function:**

**run(work_dir)**

- **Signature:**

      int run(const char* work_dir)

- **Description:**

    Initiates a flood simulation using the specified working directory. The working directory
    should contain all required input data (e.g. digital elevation models, rainfall data, land cover)
    and will be used to store the simulation outputs.

- **Parameters:**

  - **work_dir (str):**  
    The path to the working directory for simulation inputs and outputs.

- **Returns:**

  - **int:**  
    A status code where `0` indicates that the simulation completed successfully, and any non‑zero
    value signals an error.

- **Example:**

.. code-block:: python

    from synxflow import flood

    work_dir = "/path/to/flood/work_dir"
    status = flood.run(work_dir)
    if status == 0:
        print("Flood simulation completed successfully!")
    else:
        print("Flood simulation encountered an error.")

landslide
-----------------------
The **landslide** module implements the core engine for simulating landslide runout events.
It calculates the dynamics of landslides based on terrain, material properties, and environmental factors.

**Key Function:**

**run(work_dir)**

- **Signature:**

      int run(const char* work_dir)

- **Description:**

    Starts a landslide runout simulation using the provided working directory. The directory must
    include all necessary input data (such as terrain models and material properties) and serves as
    the location for simulation outputs.

- **Parameters:**

  - **work_dir (str):**  
    The path to the working directory containing the input configuration and data files.

- **Returns:**

  - **int:**  
    A status code indicating the result of the simulation (0 for success; non‑zero for an error).

- **Example:**

.. code-block:: python

    from synxflow import landslide

    work_dir = "/path/to/landslide/work_dir"
    status = landslide.run(work_dir)
    if status == 0:
        print("Landslide simulation completed successfully!")
    else:
        print("Landslide simulation encountered an error.")

debris
-----------------------------
The **debris** module provides the core engine for running debris flow simulations.

**Key Function:**

**run(work_dir)**

- **Signature:**

      int run(const char* work_dir)

- **Description:**

    Executes a debris flow simulation with the working directory specified. This directory is used
    to provide all required simulation inputs (e.g. configuration files, terrain data) and to store
    the results of the simulation.

- **Parameters:**

  - **work_dir (str):**  
    The path to the working directory where input files are stored and outputs will be generated.

- **Returns:**

  - **int:**  
    A status code that indicates the outcome of the simulation (0 indicates success; any non‑zero
    value indicates an error).

- **Example:**

.. code-block:: python

    from synxflow import debris

    work_dir = "/path/to/debris/work_dir"
    status = debris.run(work_dir)
    if status == 0:
        print("Debris flow simulation completed successfully!")
    else:
        print("Debris flow simulation encountered an error.")




flood
-----------------------
The **flood** module provides the core functionality for simulating flood inundation events.

**Key Function:**

**run(work_dir)**

- **Signature:**

      int run(const char* work_dir)

- **Description:**

    Initiates a flood simulation using the specified working directory. The working directory
    should contain all required input data (e.g. digital elevation models, rainfall data, land cover)
    and will be used to store the simulation outputs.

- **Parameters:**

  - **work_dir (str):**  
    The path to the working directory for simulation inputs and outputs.

- **Returns:**

  - **int:**  
    A status code where `0` indicates that the simulation completed successfully, and any non‑zero
    value signals an error.

- **Example:**

.. code-block:: python

    from synxflow import flood

    work_dir = "/path/to/flood/work_dir"
    status = flood.run(work_dir)
    if status == 0:
        print("Flood simulation completed successfully!")
    else:
        print("Flood simulation encountered an error.")

Landslide Runout Engine
-----------------------
The **landslide** module implements the core engine for simulating landslide runout events.

**Key Function:**

**run(work_dir)**

- **Signature:**

      int run(const char* work_dir)

- **Description:**

    Starts a landslide runout simulation using the provided working directory. The directory must
    include all necessary input data (such as terrain models and material properties) and serves as
    the location for simulation outputs.

- **Parameters:**

  - **work_dir (str):**  
    The path to the working directory containing the input configuration and data files.

- **Returns:**

  - **int:**  
    A status code indicating the result of the simulation (0 for success; non‑zero for an error).

- **Example:**

.. code-block:: python

    from synxflow import landslide

    work_dir = "/path/to/landslide/work_dir"
    status = landslide.run(work_dir)
    if status == 0:
        print("Landslide simulation completed successfully!")
    else:
        print("Landslide simulation encountered an error.")
