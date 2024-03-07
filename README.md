# WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> [![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2312.07531) <a href="https://wham.is.tue.mpg.de/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ysUtGSwidTQIdBQRhq0hj63KbseFujkn?usp=sharing)
 [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wham-reconstructing-world-grounded-humans/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=wham-reconstructing-world-grounded-humans) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wham-reconstructing-world-grounded-humans/3d-human-pose-estimation-on-emdb)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-emdb?p=wham-reconstructing-world-grounded-humans)


https://github.com/yohanshin/WHAM/assets/46889727/da4602b4-0597-4e64-8da4-ab06931b23ee


## Introduction
This repository is the official [Pytorch](https://pytorch.org/) implementation of [WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion](https://arxiv.org/abs/2312.07531). For more information, please visit our [project page](https://wham.is.tue.mpg.de/).


## Installation
Please see [Installation](docs/INSTALL.md) for details.


## Quick Demo

### [<img src="https://i.imgur.com/QCojoJk.png" width="30"> Google Colab for WHAM demo is now available](https://colab.research.google.com/drive/1ysUtGSwidTQIdBQRhq0hj63KbseFujkn?usp=sharing)

### Registration

To download SMPL body models (Neutral, Female, and Male), you need to register for [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLify](https://smplify.is.tue.mpg.de/). The username and password for both homepages will be used while fetching the demo data.

Next, run the following script to fetch demo data. This script will download all the required dependencies including trained models and demo videos.

```bash
bash fetch_demo_data.sh
```

You can try with one examplar video:
```
python demo.py --video examples/IMG_9732.mov --visualize
```

We assume camera focal length following [CLIFF](https://github.com/haofanwang/CLIFF). You can specify known camera intrinsics [fx fy cx cy] for SLAM as the demo example below:
```
python demo.py --video examples/drone_video.mp4 --calib examples/drone_calib.txt --visualize
```

You can skip SLAM if you only want to get camera-coordinate motion. You can run as:
```
python demo.py --video examples/IMG_9732.mov --visualize --estimate_local_only
```

You can further refine the results of WHAM using Temporal SMPLify as a post processing. This will allow better 2D alignment as well as 3D accuracy. All you need to do is add `--run_smplify` flag when running demo.

## Docker

Please refer to [Docker](docs/DOCKER.md) for details.

## Python API

Please refer to [API](docs/API.md) for details.

## Dataset
Please see [Dataset](docs/DATASET.md) for details.

## Evaluation
```bash
# Evaluate on 3DPW dataset
python -m lib.eval.evaluate_3dpw --cfg configs/yamls/demo.yaml TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar

# Evaluate on RICH dataset
python -m lib.eval.evaluate_rich --cfg configs/yamls/demo.yaml TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar

# Evaluate on EMDB dataset (also computes W-MPJPE and WA-MPJPE)
python -m lib.eval.evaluate_emdb --cfg configs/yamls/demo.yaml --eval-split 1 TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar   # EMDB 1

python -m lib.eval.evaluate_emdb --cfg configs/yamls/demo.yaml --eval-split 2 TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar   # EMDB 2
```

## Training
WHAM training involves into two different stages; (1) 2D to SMPL lifting through AMASS dataset and (2) finetuning with feature integration using the video datasets. Please see [Dataset](docs/DATASET.md) for preprocessing the training datasets.

### Stage 1.
```bash
python train.py --cfg configs/yamls/stage1.yaml
```

### Stage 2.
TBD

### Train with BEDLAM
TBD

## Acknowledgement
We would like to sincerely appreciate Hongwei Yi and Silvia Zuffi for the discussion and proofreading. Part of this work was done when Soyong Shin was an intern at the Max Planck Institute for Intelligence System.

The base implementation is largely borrowed from [VIBE](https://github.com/mkocabas/VIBE) and [TCMR](https://github.com/hongsukchoi/TCMR_RELEASE). We use [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) for 2D keypoints detection and [DPVO](https://github.com/princeton-vl/DPVO), [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) for extracting camera motion. Please visit their official websites for more details.

## TODO

- [ ] Training implementation

- [ ] Colab / Hugging face release

- [x] Demo for custom videos

## Citation
```
@article{shin2023wham,
    title={WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion},
    author={Shin, Soyong and Kim, Juyong and Halilaj, Eni and Black, Michael J.},
    journal={arXiv preprint 2312.07531},
    year={2023}}
```

## License
Please see [License](./LICENSE) for details.

## Contact
Please contact soyongs@andrew.cmu.edu for any questions related to this work.
