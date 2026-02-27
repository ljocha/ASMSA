# ASMSA
Analysis and Sampling of Molecular Simulations by adversarial Autoencoders

Use `nbstrioput` to commit so that diffing/merging is not nightmare.

## Local installation

Find a reasonable (4+ CPU cores, 32+ GB RAM, GPU welcome) Linux machine,
preferably with recent stable OS release (Ubuntu 24.04, Debian 13, ...)

### GPU support

Not strictly required but things will be slooooooow otherwise ...

Things work with Nvidia/cuda now, will play with AMD/rocm and Apple/Metal one day ...

So make sure decent Nvidia GPU is installed and follow installation instructions 
for your Linux flavor.
If drivers find it, `nvidia-smi` tells you.

### Docker 

We run Gromacs from our Docker image, hence Docker is needed.

- Install according to the instructions for your OS: https://docs.docker.com/engine/install/
- Install Nvidia container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

Check with `docker run --rm --gpus all ubuntu nvidia-smi`, it should say the same as plain `nvidia-smi`.
Troubleshoot by asking your favourite AI tool, they are fairly good in it.

### Conda

Though Conda is my least prefered package manager, we want to be compatible with Binder environment etc.

Install Miniforge: https://github.com/conda-forge/miniforge (or some other distribution which will provide `mamba` command
and access to `conda-forge` channel.

### Create and install ASMSA into virtual environment

Run `make install_local` from this directory. It takes few minutes to download all the required packages.

### Run from the virtual environment

Run `make run_local` -- besides activating the environment it prepends PATH to have the right `gmx` command, and maybe other specific settings.

Jupyterlab starts, the command yiedls URL to go to. In this setup, we don't complicate things with SSL, the server runs on localhost only; if it is a remote machine, one has to use ssh port forwarding etc.


## Run in MDDash

TODO
