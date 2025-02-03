==========================================
Simulation Engines
==========================================

The SynxFlow package includes multiple simulation engine modules implemented as CUDA extensions.
Each engine is exposed as a Python module with a common interface. The following sections describe
the API for each simulation engine:

- **flood**: Flood simulation engine.
- **landslide**: Landslide runout engine.
- **debris**: Debris flow simulation engine.

.. py:module:: flood

   **Overview:**

   The *flood* module provides the core functionality for simulating flood dynamics using GPU acceleration.

   **Function: run(work_dir)**

   .. py:function:: run(work_dir)

      Executes the flood simulation.

      :param work_dir: str
         The path to the working directory that contains all necessary input files (e.g. digital elevation models, rainfall data)
         and where simulation outputs will be stored.
      :returns: int
         A status code where a return value of 0 indicates that the simulation completed successfully,
         and any non-zero value indicates an error.

   .. py:function:: run_mgpus(work_dir)

      Executes the flood simulation using multiple GPUs.

      :param work_dir: str
         The path to the working directory that contains all necessary input files (e.g. digital elevation models, rainfall data)
         and where simulation outputs will be stored.
      :returns: int
         A status code where a return value of 0 indicates that the simulation completed successfully,
         and any non-zero value indicates an error.

   **Example:**

   .. code-block:: python

      from synxflow import flood

      status = flood.run("/path/to/flood/work_dir")
      if status == 0:
          print("Flood simulation completed successfully!")
      else:
          print("Flood simulation encountered an error.")


.. py:module:: landslide

   **Overview:**

   The *landslide* module provides the core functionality for simulating landslide runout events using GPU acceleration.

   **Function: run(work_dir)**

   .. py:function:: run(work_dir)

      Starts the landslide runout simulation.

      :param work_dir: str
         The path to the working directory containing the necessary input data (e.g. terrain models, material properties)
         for the landslide simulation, and where outputs will be written.
      :returns: int
         A status code where 0 indicates success and a non-zero value indicates an error.

   **Example:**

   .. code-block:: python

      from synxflow import landslide

      status = landslide.run("/path/to/landslide/work_dir")
      if status == 0:
          print("Landslide simulation completed successfully!")
      else:
          print("Landslide simulation encountered an error.")


.. py:module:: debris

   **Overview:**

   The *debris* module provides the core functionality for running debris flow simulations using GPU acceleration.
   It is optimized for high-performance computation and is designed for multi-hazard risk assessments.

   **Function: run(work_dir)**

   .. py:function:: run(work_dir)

      Executes the debris flow simulation.

      :param work_dir: str
         The working directory containing simulation input files (e.g. configuration files, terrain data)
         and where the simulation outputs will be generated.
      :returns: int
         A status code indicating the result of the simulation. Zero denotes a successful simulation,
         whereas any non-zero value indicates an error.

   **Example:**

   .. code-block:: python

      from synxflow import debris

      status = debris.run("/path/to/debris/work_dir")
      if status == 0:
          print("Debris flow simulation completed successfully!")
      else:
          print("Debris flow simulation encountered an error!")
