# Author: Thivin Anandh
# Dockerfile to build a container image for SynxFlow

# Use NVIDIA CUDA 11.8 base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Avoid tzdata interactive configuration
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    software-properties-common \
    vim
    # remove the default python3 symlink

# Add deadsnakes PPA to get python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa

# Update package list again to include packages from the PPA
RUN apt-get update

# Install Python 3.11 (or your desired version)
RUN apt-get install -y python3.11 python3.11-dev python3.11-distutils python3.11-venv

# Set Cmake variables to use the correct python version
ENV CMAKE_PYTHON_EXECUTABLE=/usr/bin/python3.11
ENV PATH=/usr/bin/python3.11:${PATH}
ENV CMAKE_ARGS="-DCMAKE_PYTHON_EXECUTABLE=/usr/bin/python3.11"

# Setup a venv so that the cmake build can find the correct python version
RUN python3.11 -m venv /venv
ENV PATH=/venv/bin:${PATH}

# activate the venv
RUN . /venv/bin/activate

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# install Jupyter-Lab
RUN python3.11 -m pip install jupyterlab

# Set environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV CUDAToolkit_ROOT=${CUDA_HOME}

# Install the package
RUN python3.11 -m pip install git+https://github.com/SynxFlow/SynxFlow.git

# Verify installation
RUN echo '#!/bin/bash\n\
python3.11 -c "\
try:\n\
    import synxflow\n\
    import synxflow.IO\n\
    import synxflow.flood\n\
    print(\\"\\\\n===================================\\");\n\
    print(\\"SynxFlow Installation Successful!\\");\n\
    print(\\"===================================\\\\n\\");\n\
except Exception as e:\n\
    print(\\"Installation Failed!\\");\n\
    print(e);\n\
    exit(1);\n\
"' > /verify_install.sh && \
    chmod +x /verify_install.sh

# Create a new script that runs verification and then bash
RUN echo '#!/bin/bash\n\
source /venv/bin/activate\n\
/verify_install.sh\n\
exec /bin/bash\n'\
> /entrypoint.sh && \
chmod +x /entrypoint.sh

# Use the new script as entrypoint
ENTRYPOINT ["/entrypoint.sh"]
