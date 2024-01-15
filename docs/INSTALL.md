# Installation

WHAM has been implemented and tested on Ubuntu 20.04 and 22.04 with python = 3.9. We provide [anaconda](https://www.anaconda.com/) environment to run WHAM as below.

```bash
# Clone the repo
git clone https://github.com/yohanshin/WHAM.git --recursive
cd WHAM/

# Create Conda environment
conda create -n wham python=3.9
conda activate wham

# Install PyTorch libraries
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Install PyTorch3D (optional) for visualization
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html

# Install WHAM dependencies
pip install -r requirements.txt

# Install ViTPose
pip install -v -e third-party/ViTPose

# Install DPVO
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
conda install pytorch-scatter=2.0.9 -c rusty1s
conda install cudatoolkit-dev=11.3.1 -c conda-forge

# ONLY IF your GCC version is larger than 10
conda install -c conda-forge gxx=9.5

pip install .
```
