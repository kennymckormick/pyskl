# MSG3D

## Abstract

Spatial-temporal graphs have been widely used by skeleton-based action recognition algorithms to model human action dynamics. To capture robust movement patterns from these graphs, long-range and multi-scale context aggregation and spatial-temporal dependency modeling are critical aspects of a powerful feature extractor. However, existing methods have limitations in achieving (1) unbiased long-range joint relationship modeling under multi-scale operators and (2) unobstructed cross-spacetime information flow for capturing complex spatial-temporal dependencies. In this work, we present (1) a simple method to disentangle multi-scale graph convolutions and (2) a unified spatial-temporal graph convolutional operator named G3D. The proposed multi-scale aggregation scheme disentangles the importance of nodes in different neighborhoods for effective long-range modeling. The proposed G3D module leverages dense cross-spacetime edges as skip connections for direct information propagation across the spatial-temporal graph. By coupling these proposals, we develop a powerful feature extractor named MS-G3D based on which our model outperforms previous state-of-the-art methods on three large-scale datasets: NTU RGB+D 60, NTU RGB+D 120, and Kinetics Skeleton 400.

## Citation

```BibTeX
@inproceedings{liu2020disentangling,
  title={Disentangling and unifying graph convolutions for skeleton-based action recognition},
  author={Liu, Ziyu and Zhang, Hongwen and Chen, Zhenghao and Wang, Zhiyong and Ouyang, Wanli},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={143--152},
  year={2020}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | GPUs | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone-Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | 8 | [joint_config](/configs/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/j.py): [89.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/b.py): [89.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/jm.py): [87.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/bm.py): [86.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/bm.pth) | 91.0 | 91.7 |
| NTURGB+D XSub | HRNet 2D Skeleton | 8 | [joint_config](/configs/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/j.py): [92.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/j.pth) | [bone_config](/configs/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/b.py): [92.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/b.pth) | [joint_motion_config](/configs/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/jm.py): [89.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/bm.py): [90.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xsub_hrnet/bm.pth) | 93.8 | 94.1 |
| NTURGB+D XView | Official 3D Skeleton | 8 | [joint_config](/configs/msg3d/msg3d_pyskl_ntu60_xview_3dkp/j.py): [95.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xview_3dkp/j.pth) | [bone_config](/configs/msg3d/msg3d_pyskl_ntu60_xview_3dkp/b.py): [95.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xview_3dkp/b.pth) | [joint_motion_config](/configs/msg3d/msg3d_pyskl_ntu60_xview_3dkp/jm.py): [94.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/msg3d/msg3d_pyskl_ntu60_xview_3dkp/bm.py): [92.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xview_3dkp/bm.pth) | 96.4 | 96.9 |
| NTURGB+D XView | HRNet 2D Skeleton | 8 | [joint_config](/configs/msg3d/msg3d_pyskl_ntu60_xview_hrnet/j.py): [97.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xview_hrnet/j.pth) | [bone_config](/configs/msg3d/msg3d_pyskl_ntu60_xview_hrnet/b.py): [97.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xview_hrnet/b.pth) | [joint_motion_config](/configs/msg3d/msg3d_pyskl_ntu60_xview_hrnet/jm.py): [95.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xview_hrnet/jm.pth) | [bone_motion_config](/configs/msg3d/msg3d_pyskl_ntu60_xview_hrnet/bm.py): [95.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu60_xview_hrnet/bm.pth) | 97.9 | 98.3 |
| NTURGB+D 120 XSub | Official 3D Skeleton | 8 | [joint_config](/configs/msg3d/msg3d_pyskl_ntu120_xsub_3dkp/j.py): [84.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xsub_3dkp/j.pth) | [bone_config](/configs/msg3d/msg3d_pyskl_ntu120_xsub_3dkp/b.py): [85.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xsub_3dkp/b.pth) | [joint_motion_config](/configs/msg3d/msg3d_pyskl_ntu120_xsub_3dkp/jm.py): [82.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/msg3d/msg3d_pyskl_ntu120_xsub_3dkp/bm.py): [81.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xsub_3dkp/bm.pth) | 86.9 | 87.8 |
| NTURGB+D 120 XSub | HRNet 2D Skeleton | 8 | [joint_config](/configs/msg3d/msg3d_pyskl_ntu120_xsub_hrnet/j.py): [85.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xsub_hrnet/j.pth) | [bone_config](/configs/msg3d/msg3d_pyskl_ntu120_xsub_hrnet/b.py): [85.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xsub_hrnet/b.pth) | [joint_motion_config](/configs/msg3d/msg3d_pyskl_ntu120_xsub_hrnet/jm.py): [82.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/msg3d/msg3d_pyskl_ntu120_xsub_hrnet/bm.py): [82.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xsub_hrnet/bm.pth) | 86.7 | 87.4 |
| NTURGB+D 120 XSet | Official 3D Skeleton | 8 | [joint_config](/configs/msg3d/msg3d_pyskl_ntu120_xset_3dkp/j.py): [86.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xset_3dkp/j.pth) | [bone_config](/configs/msg3d/msg3d_pyskl_ntu120_xset_3dkp/b.py): [87.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xset_3dkp/b.pth) | [joint_motion_config](/configs/msg3d/msg3d_pyskl_ntu120_xset_3dkp/jm.py): [82.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xset_3dkp/jm.pth) | [bone_motion_config](/configs/msg3d/msg3d_pyskl_ntu120_xset_3dkp/bm.py): [83.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xset_3dkp/bm.pth) | 88.9 | 89.6 |
| NTURGB+D 120 XSet | HRNet 2D Skeleton | 8 | [joint_config](/configs/msg3d/msg3d_pyskl_ntu120_xset_hrnet/j.py): [88.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xset_hrnet/j.pth) | [bone_config](/configs/msg3d/msg3d_pyskl_ntu120_xset_hrnet/b.py): [88.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xset_hrnet/b.pth) | [joint_motion_config](/configs/msg3d/msg3d_pyskl_ntu120_xset_hrnet/jm.py): [86.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xset_hrnet/jm.pth) | [bone_motion_config](/configs/msg3d/msg3d_pyskl_ntu120_xset_hrnet/bm.py): [86.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/msg3d/msg3d_pyskl_ntu120_xset_hrnet/bm.pth) | 90.0 | 90.9 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train MSG3D on NTURGB+D XSub (3D skeleton, Joint Modality) with 8 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/j.py 8 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test MSG3D on NTURGB+D XSub (3D skeleton, Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/msg3d/msg3d_pyskl_ntu60_xsub_3dkp/j.py checkpoints/SOME_CHECKPOINT.pth 8 --eval top_k_accuracy --out result.pkl
```
