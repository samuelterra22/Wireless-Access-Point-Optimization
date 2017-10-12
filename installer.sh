#!/usr/bin/env bash

################################################################################
## SOFTWARE https://github.com/samuelterra22/tcc/
wget -c https://codeload.github.com/samuelterra22/tcc/zip/master -o tcc-master.zip
unzip tcc-master.zip

################################################################################
## PYTHON libs
sudo apt update
sudo apt install python-pip && sudo -H pip install --upgrade pip
sudo -H pip install ezdxf numpy numba matplotlib pygame colour datetime
sudo -H pip install tk || sudo apt install python-tk

################################################################################
## CUDA
wget -c http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

sudo apt-get update

sudo apt-get install cuda

sudo sed -i 's/"$/:\/usr\/local\/cuda-8.0\/bin"/' /etc/environment
source /etc/environment

nvcc --version
nvidia-smi

## HOWTO: http://www.pradeepadiga.me/blog/2017/03/22/installing-cuda-toolkit-8-0-on-ubuntu-16-04/

################################################################################
## TEST
python CAD/draw.py

################################################################################
## RUN
python PlacementGPU.py