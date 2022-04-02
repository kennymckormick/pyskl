# PYSKL

PYSKL is a toolbox focusing on action recognition based on **SK**e**L**eton data with **PY**Torch. Various algorithms will be supported for skeleton-based action recognition. We build this project based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2).

This repo is the official implementation of [PoseConv3D](https://arxiv.org/abs/2104.13586) and [STGCN++]().

## Supported Algorithms

- [x] ST-GCN (AAAI 2018): https://arxiv.org/abs/1801.07455 [[MODELZOO](/configs/stgcn/README.md)]
- [x] ST-GCN++ (PYSKL): [Tech Report Coming Soon]() [[MODELZOO](/configs/stgcn++/README.md)]
- [x] PoseConv3D (CVPR 2022 Oral): https://arxiv.org/abs/2104.13586 [[MODELZOO](/configs/posec3d/README.md)]

## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
pip install -r requirements.txt
pip install -e .
```

## Data Preparation
For data pre-processing, we estimate 2D skeletons with a two-stage pose estimator (Faster-RCNN + HRNet). For 3D skeletons, we follow the pre-processing procedure of [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). Currently, we do not provide the pre-processing scripts. Instead, we directly provide the [processed skeleton data](/tools/data_list.md) as pickle files, which can be directly used in training and evaluation.  You can use [vis_skeleton](/demo/vis_skeleton.ipynb) to visualize the provided skeleton data.


## Training & Testing
You can use following commands for training and testing. Basically, we support distribued training on a single server with multiple GPUs.
```shell
# Training
bash tools/dist_train.sh {config_name} {num_gpus} {other_options}
# Testing
bash tools/dist_test.sh {config_name} {checkpoint} {num_gpus} --out {output_file} --eval top_k_accuracy mean_class_accuracy
```
For specific examples, please go to the README for each specific algorithm we supported.

## Citation

If you use PYSKL in your research or wish to refer to the baseline results published in the Model Zoo, please use the following BibTeX entry and the BibTex entry corresponding to the specific algorithm you used.

```BibTeX
% Tech Report Coming Soon!
@misc{duan2022pyskl,
    title={PYSKL: a toolbox for skeleton-based video understanding},
    author={PYSKL Contributors},
    howpublished = {\url{https://github.com/kennymckormick/pyskl}},
    year={2022}
}
```
