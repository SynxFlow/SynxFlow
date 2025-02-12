# SynxFlow Docker Environment
---


This repository provides a Docker environment for running SynxFlow with CUDA support and JupyterLab integration.

## Prerequisites
---

### Software Requirements
1. NVIDIA GPU Drivers
   ```bash
   # Check if NVIDIA drivers are installed
   nvidia-smi
   ```

2. Docker
 Follow the instructions [here](https://docs.docker.com/engine/install/ubuntu/) to install Docker.

3. NVIDIA Container Toolkit
   ```bash
   # Install NVIDIA Container Toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```
For more information, refer to the official [NVIDIA documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Building the Docker Image

```bash
# Clone this repository
git clone https://github.com/SynxFlow/SynxFlow.git
cd SynxFlow

# Build the Docker image
docker build -t synxflow .
```

## Running the Container

### Local Usage
```bash
# Run container with GPU support
docker run --gpus all -it -p 8888:8888 synxflow
```

If you need to run the juptyer lab on the docker container, you can run the following command:
```bash

# Inside the container, start JupyterLab
source /venv/bin/activate && jupyter lab --ip 0.0.0.0 --port 8888 --allow-root --no-browser
```
