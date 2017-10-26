#!/usr/bin/env bash

########################################################################################################################
## SOFTWARE https://github.com/samuelterra22/tcc/

notify-send 'Obtendo projeto' 'Obtendo projeto mais atualizado do GitHub.' --icon=dialog-information
echo "Obtendo projeto mais atualizado do GitHub."
wget -c https://codeload.github.com/samuelterra22/tcc/zip/master -o tcc-master.zip
unzip tcc-master.zip

########################################################################################################################
## PYTHON libs

notify-send 'Instalando dependências' 'Iniciando instalação de dependências para a execução do Placement.' --icon=dialog-information
sudo apt-get update
sudo apt-get install -y build-essential llvm libsdl1.2-dev libglew1.5-dev freeglut3-dev mesa-common-dev
sudo apt-get -y install python-pip && sudo -H pip install --upgrade pip
sudo -H pip install ezdxf numpy numba matplotlib pygame colour datetime llvmpy
sudo -H pip install tk || sudo apt install python-tk

########################################################################################################################
## CUDA

notify-send 'Obtendo Cuda' 'Iniciando download do CUDA. O arquivo de instalação tem aproximadamente 1,2 GB. O processo de download pode levar alguns minutos.' --icon=dialog-information
echo "Iniciando download do CUDA. O arquivo de instalação tem aproximadamente 1,2 GB. O processo de download pode levar alguns minutos."
wget -c https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb

notify-send 'Instalando Cuda' 'Iniciando instalação do CUDA apartir do arquivo baixado.' --icon=dialog-information
echo "Iniciando instalação do CUDA apartir do arquivo baixado."
sudo dpkg -i cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64.deb

sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
#sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub

sudo apt-get update

sudo apt-get -y install -f

sudo apt-get -y install cuda cuda-9-0 cuda-toolkit-9-0 cuda-runtime-9-0 cuda-libraries-9-0 cuda-libraries-dev-9-0 cuda-drivers

sudo sed -i 's/"$/:\/usr\/local\/cuda-8.0\/bin"/' /etc/environment
source /etc/environment

notify-send 'Instalação finalizada.' 'A instalação do CUDA foi finalizada.' --icon=dialog-information
echo "A instalação do CUDA foi finalizada."
echo "Versão do nvcc:\n"
nvcc --version
echo "\nResumo do driver instalado:\n"
nvidia-smi

## HOWTO: http://www.pradeepadiga.me/blog/2017/03/22/installing-cuda-toolkit-8-0-on-ubuntu-16-04/
## HOWTO: https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1604&target_type=deblocal

########################################################################################################################
## TEST
python CAD/draw.py

########################################################################################################################
## RUN
#python PlacementGPU.py
python PlacementAPs.py


