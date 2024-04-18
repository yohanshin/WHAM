# Dataset

## Training Data
We use [AMASS](https://amass.is.tue.mpg.de/), [InstaVariety](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md), [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/), [Human3.6M](http://vision.imar.ro/human3.6m/description.php), and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) datasets for training. Please register to their websites to download and process the data. You can download parsed ViT version of InstaVariety, MPI-INF-3DHP, Human3.6M, and 3DPW data from the [Google Drive](https://drive.google.com/drive/folders/13T2ghVvrw_fEk3X-8L0e6DVSYx_Og8o3?usp=sharing). You can save the data under `dataset/parsed_data` folder.

### Process AMASS dataset
After downloading AMASS dataset, you can process it by running:
```bash
python -m lib.data_utils.amass_utils
```
The processed data will be stored at `dataset/parsed_data/amass.pth`.

### Process 3DPW, MPII3D, Human3.6M, and InstaVariety datasets
First, visit [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE) and download preprocessed data at `dataset/parsed_data/TCMR_preproc/'.

Next, prepare 2D keypoints detection using [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and store the results at `dataset/detection_results/\<DATAsET-NAME>/\<SEQUENCE_NAME.npy>'. You may need to download all images to prepare the detection results.

For Human36M, MPII3D, and InstaVariety datasets, you need to also download [NeuralAnnot](https://github.com/mks0601/NeuralAnnot_RELEASE) pseudo groundtruth SMPL label. As mentioned in our paper, we do not supervise WHAM on this label, but use it for neural initialization step.

Finally, run following codes to preprocess all training data.
```bash
python -m lib.data_utils.threedpw_train_utils       # 3DPW dataset
# [Coming] python -m lib.data_utils.human36m_train_utils       # Human3.6M dataset
# [Coming] python -m lib.data_utils.mpii3d_train_utils         # MPI-INF-3DHP dataset
# [Coming] python -m lib.data_utils.insta_train_utils          # InstaVariety dataset
```

### Process BEDLAM dataset
Will be updated.

## Evaluation Data
We use [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/), [RICH](https://rich.is.tue.mpg.de/), and [EMDB](https://eth-ait.github.io/emdb/) for the evaluation. We provide the parsed data for the evaluation. Please download the data from [Google Drive](https://drive.google.com/drive/folders/13T2ghVvrw_fEk3X-8L0e6DVSYx_Og8o3?usp=sharing) and place them at `dataset/parsed_data/`.

To process the data at your end, please 
1) Download parsed 3DPW data from [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE) and store `dataset/parsed_data/TCMR_preproc/'.
2) Run [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) on all test data and store the results at `dataset/detection_results/\<DATAsET-NAME>'.
3) Run following codes.
```bash
python -m lib.data_utils.threedpw_eval_utils --split <"val" or "test">      # 3DPW dataset
python -m lib.data_utils.emdb_eval_utils --split <"1" or "2">               # EMDB dataset
python -m lib.data_utils.rich_eval_utils                                    # RICH dataset
```