export CONDA_ENV_NAME=wham
echo 'Start Anaconda installation. Environment name:' $CONDA_ENV_NAME

# Create Conda environment
conda create -y -n $CONDA_ENV_NAME python=3.9
conda activate $CONDA_ENV_NAME

which python
which pip

# Install PyTorch libraries
echo "=========> Installing PyTorch ..."
conda -y install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

# Install PyTorch3D (optional) for visualization
echo "=========> Installing PyTorch3D ..."
conda -y install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1110/download.html

# Install WHAM dependencies
echo "=========> Installing Dependencies ..."
pip install -r requirements.txt

# Install ViTPose
echo "=========> Installing ViTPose and ViTDet ..."
pip install -v -e third-party/ViTPose
pip install -v -e third-party/ViTDet

# Install DPVO
echo "=========> Installing DPVO ..."
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
conda -y install pytorch-scatter=2.0.9 -c rusty1s
conda -y install cudatoolkit-dev=11.3.1 -c conda-forge

# ONLY IF your GCC version is larger than 10
conda -y install -c conda-forge gxx=9.5

pip install .
echo "Completed Installation !"

cd ../../