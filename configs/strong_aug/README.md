# Strong Augmentations

## Introduction

This directory includes configs for training ST-GCN++ with strong spatial augmentations and 120 epochs. The augmentations we adopted include Random Rotating and Random Scaling.

## Citation

```BibTeX
@misc{duan2022pyskl,
    title={PYSKL: a toolbox for skeleton-based video understanding},
    author={PYSKL Contributors},
    howpublished = {\url{https://github.com/kennymckormick/pyskl}},
    year={2022}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | GPUs | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone-Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | 8 | [joint_config](/configs/strong_aug/ntu60_xsub_3dkp/j.py): [90.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/strong_aug/ntu60_xsub_3dkp/b.py): [90.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/strong_aug/ntu60_xsub_3dkp/jm.py): [88.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/strong_aug/ntu60_xsub_3dkp/bm.py): [87.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xsub_3dkp/bm.pth) | 92.2 | 92.6 |
| NTURGB+D XView | Official 3D Skeleton | 8 | [joint_config](/configs/strong_aug/ntu60_xview_3dkp/j.py): [96.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xview_3dkp/j.pth) | [bone_config](/configs/strong_aug/ntu60_xview_3dkp/b.py): [95.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xview_3dkp/b.pth) | [joint_motion_config](/configs/strong_aug/ntu60_xview_3dkp/jm.py): [95.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/strong_aug/ntu60_xview_3dkp/bm.py): [93.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu60_xview_3dkp/bm.pth) | 97.1 | 97.4 |
| NTURGB+D 120 XSub | Official 3D Skeleton | 8 | [joint_config](/configs/strong_aug/ntu120_xsub_3dkp/j.py): [84.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xsub_3dkp/j.pth) | [bone_config](/configs/strong_aug/ntu120_xsub_3dkp/b.py): [87.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xsub_3dkp/b.pth) | [joint_motion_config](/configs/strong_aug/ntu120_xsub_3dkp/jm.py): [82.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/strong_aug/ntu120_xsub_3dkp/bm.py): [81.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xsub_3dkp/bm.pth) | 88.2 | 88.6 |
| NTURGB+D 120 XSet | Official 3D Skeleton | 8 | [joint_config](/configs/strong_aug/ntu120_xset_3dkp/j.py): [86.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xset_3dkp/j.pth) | [bone_config](/configs/strong_aug/ntu120_xset_3dkp/b.py): [88.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xset_3dkp/b.pth) | [joint_motion_config](/configs/strong_aug/ntu120_xset_3dkp/jm.py): [85.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xset_3dkp/jm.pth) | [bone_motion_config](/configs/strong_aug/ntu120_xset_3dkp/bm.py): [84.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/strong_aug/ntu120_xset_3dkp/bm.pth) | 90.1 | 90.8 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion.


## Training & Testing

Please refer to the README of ST-GCN++.
