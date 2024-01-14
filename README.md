# WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> [![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2312.07531) <a href="https://wham.is.tue.mpg.de/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a> [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wham-reconstructing-world-grounded-humans/3d-human-pose-estimation-on-3dpw)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-3dpw?p=wham-reconstructing-world-grounded-humans) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/wham-reconstructing-world-grounded-humans/3d-human-pose-estimation-on-emdb)](https://paperswithcode.com/sota/3d-human-pose-estimation-on-emdb?p=wham-reconstructing-world-grounded-humans)


https://github.com/yohanshin/WHAM/assets/46889727/da4602b4-0597-4e64-8da4-ab06931b23ee


## Introduction
This repository is the official [Pytorch](https://pytorch.org/) implementation of [WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion](https://arxiv.org/abs/2312.07531). For more information, please visit our [project page](https://wham.is.tue.mpg.de/).


## Installation
Please see [Installation](docs/INSTALL.md) for details.


## Quick Demo
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

## Dataset
Please see [Dataset](docs/DATASET.md) for details.



## Evaluation
```
# Evaluate on 3DPW dataset
python -m lib.eval.evaluate_3dpw --cfg configs/yamls/demo.yaml TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar

# Evaluate on RICH dataset
python -m lib.eval.evaluate_rich --cfg configs/yamls/demo.yaml TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar

# Evaluate on EMDB dataset (also computes W-MPJPE and WA-MPJPE)
python -m lib.eval.evaluate_emdb --cfg configs/yamls/demo.yaml --eval-split 1 TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar   # EMDB 1

python -m lib.eval.evaluate_emdb --cfg configs/yamls/demo.yaml --eval-split 2 TRAIN.CHECKPOINT checkpoints/wham_vit_w_3dpw.pth.tar   # EMDB 2
```

## Training
Will be updated.

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
