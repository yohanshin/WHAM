# Installation

WHAM has been implemented and tested on Ubuntu 20.04 with python = 3.9. We provide [anaconda](https://www.anaconda.com/) environment to run WHAM as below.

```bash
# Clone the repo
git clone https://github.com/yohanshin/WHAM.git

# Create Conda environment
conda create -n wham python=3.9
conda activate wham

# Install PyTorch libraries
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install PyTorch3D (optional) for visualization
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu113_pyt1100/download.html

# Install remaining dependencies
pip install -r requirements.txt
```

Once you install all dependencies, you may need to delete the following line at `<YOUR-CONDA-PTH>/envs/mifi/lib/python3.9/site-packages/chumpy/__init__.py:line 8` that starts with `from numpy import ...`.