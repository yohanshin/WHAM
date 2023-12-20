# Dataset

## Training Data
We use [AMASS](https://amass.is.tue.mpg.de/), [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/), [Human3.6M](http://vision.imar.ro/human3.6m/description.php), and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) datasets for training. Please register to their websites to download and process the data.

More details for the training data will be updated.

## Evaluation Data
We use [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/), [RICH](https://rich.is.tue.mpg.de/), and [EMDB](https://eth-ait.github.io/emdb/) for the evaluation. We provide the parsed data for the evaluation. Please download the data from [Google Drive](https://drive.google.com/drive/folders/13T2ghVvrw_fEk3X-8L0e6DVSYx_Og8o3?usp=sharing) and place them at `dataset/parsed_data/`.

## Others
To run WHAM, you need [SMPL](https://smpl.is.tue.mpg.de/) body model and auxiliary files such as joint regression matrices. Please visit [SMPL](https://smpl.is.tue.mpg.de/) website and download the latest version of the body model. The auxiliary files can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1LDMzLxv2ImCFwVZM_aWHlTSaX311gWP-?usp=sharing). Please locate files at `dataset/body_models/`.

## Folder structure
After download and processed all data, you will have the folder structure as below.

```bash
<repo root>/
|- dataset/
      |- body_models/
      |  |- J_regressor_wham.npy
      |  |- ...
      |  |- smpl/
      |      |- SMPL_NEUTRAL.pkl
      |      |- ...
      |
      |- parsed_data/
      |      |- 3dpw_test_vit.pth
      |      |- ...
      |
      |- AMASS/
      |- 3DPW/
      |- EMDB/
      |- ...

|- checkpoints/
      |- wham_vit_w_3dpw.pth.tar
```