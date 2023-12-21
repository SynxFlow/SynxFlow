## About

### SynxFlow: Synergising High-Performance Hazard Simulation with Data Flow

SynxFlow is an open-source model capable of dynamically simulating flood inundation, landslide runout, and debris flows using CUDA-enabled GPUs. It features a user-friendly yet versatile Python interface and seamlessly integrates with tools such as Numpy, Pandas, and GDAL for efficient processing of inputs, simulation execution, and output visualisation. With the acceleration of modern GPUs, SynxFlow can complete large-scale simulations in just minutes, making it an ideal solution for integrating into data science workflows to streamline and accelerate natural hazard risk assessment, empowering both research and business applications.

### Frequently Asked Questions

#### How easy is it to use the model?

Thinking about the complexities of software installation, data preparation, or the need for powerful computing resources? With SynxFlow, running a flood model is easier than ever. Install the model with a simple command 'pip install synxflow', and set up a comprehensive flood or debris flow model with just a few lines of Python code. Plus, SynxFlow's compatibility with cloud platforms like Google Colab means you don't need a high-end computer. Check out our demonstrator to see how you can run a flood simulation in the cloud. Just click below and select 'Runtime -> Run All'.

[![Run a flood model](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ujrdzuEeFOZ1L_iETmu4G97HiZQpOb2o?usp=sharing#scrollTo=ZLFGmuK26M0v)


#### How solid is the science behind the model?

The SynxFlow authors have many years' experiences of developing and applying hydrodynamic models. The algothims used in the model are highly efficient and robust, using Godonov-type shock-capturing metheds for solving shallow water based equations. The numerical methods in the model are documented in the following papers.

[Reference for the flood model](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2016WR020055)

[Reference for the landslide runout model](https://www.sciencedirect.com/science/article/pii/S0013795217306324)

[Reference for the debris flow model](https://www.sciencedirect.com/science/article/pii/S0013795223003289)

#### What if I have trouble using the model?

If you encounter any issues or have questions, feel free to leave a comment on our [Github](https://github.com/SynxFlow/SynxFlow/discussions) discussions page. We are always happy to answer any questions.


