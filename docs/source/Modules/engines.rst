==========================================
Simulation Engines
==========================================

The SynxFlow package includes multiple simulation engines implemented as C extensions.
Each engine is exposed as a Python class with a common interface. The following sections describe
the API for each simulation engine:

- **flood**: Flood simulation engine.
- **landslide**: Landslide runout engine.
- **debris**: Debris flow simulation engine.

.. py:class:: flood(work_dir)

   :param work_dir: str
      The path to the working directory that contains all necessary input files (e.g. digital elevation models, rainfall data)
      and where simulation outputs will be stored.

   **Overview:**

   The *FloodEngine* class provides the core functionality for simulating flood dynamics using GPU acceleration.
   It performs the simulation based on the provided input data and returns a status code indicating success or failure.

   **Methods:**

   .. py:method:: run()
      
      Executes the flood simulation.

      **Returns:**

         int  
         A status code where a return value of 0 indicates the simulation completed successfully,
         and any non-zero value indicates an error.

   **Example:**

   .. code-block:: python

      from synxflow import flood

      # Create an instance of the FloodEngine with the specified working directory
      engine = flood.FloodEngine("/path/to/flood/work_dir")
      status = engine.run()
      if status == 0:
          print("Flood simulation completed successfully!")
      else:
          print("Flood simulation encountered an error.")


.. py:class:: landslide(work_dir)

   :param work_dir: str
      The path to the working directory containing the necessary input data (e.g. terrain models, material properties)
      for the landslide simulation, and where outputs will be written.

   **Overview:**

   The *LandslideEngine* class implements the core engine for simulating landslide runout events.
   It utilizes GPU acceleration to compute the dynamics of landslide movement based on input configurations.

   **Methods:**

   .. py:method:: run()
      
      Starts the landslide runout simulation.

      **Returns:**

         int  
         A status code where 0 indicates success and a non-zero value indicates an error.

   **Example:**

   .. code-block:: python

      from synxflow import landslide

      engine = landslide.LandslideEngine("/path/to/landslide/work_dir")
      status = engine.run()
      if status == 0:
          print("Landslide simulation completed successfully!")
      else:
          print("Landslide simulation encountered an error.")


.. py:class:: debris(work_dir)

   :param work_dir: str
      The working directory containing simulation input files (e.g. configuration files, terrain data) and where the simulation outputs
      will be generated.

   **Overview:**

   The *DebrisEngine* class provides the functionality to run debris flow simulations using GPU acceleration.
   It is optimized for high-performance computation and is designed to be used in multi-hazard risk assessments.

   **Methods:**

   .. py:method:: run()
      
      Executes the debris flow simulation.

      **Returns:**

         int  
         A status code indicating the result of the simulation. Zero denotes a successful simulation, whereas any non-zero value
         indicates an error.

   **Example:**

   .. code-block:: python

      from synxflow import debris

      engine = debris.DebrisEngine("/path/to/debris/work_dir")
      status = engine.run()
      if status == 0:
          print("Debris flow simulation completed successfully!")
      else:
          print("Debris flow simulation encountered an error!")

