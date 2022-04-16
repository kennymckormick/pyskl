# STGCN

## Introduction

STGCN is one of the first algorithms that adopt Graph Convolution Neural Networks for skeleton processing. We provide STGCN trained on NTURGB+D with 2D skeletons (HRNet) and 3D skeletons in both the original training setting and the **PYSKL** training setting. We provide checkpoints for four modalities: Joint, Bone, Joint Motion, and Bone Motion. The accuracy of each modality links to the weight file.

## Citation

```BibTeX
@inproceedings{yan2018spatial,
  title={Spatial temporal graph convolutional networks for skeleton-based action recognition},
  author={Yan, Sijie and Xiong, Yuanjun and Lin, Dahua},
  booktitle={Thirty-second AAAI conference on artificial intelligence},
  year={2018}
}
# If you use the STGCN with PYSKL practices in your work
@misc{duan2022pyskl,
    title={PYSKL: a toolbox for skeleton-based video understanding},
    author={PYSKL Contributors},
    howpublished = {\url{https://github.com/kennymckormick/pyskl}},
    year={2022}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Practice | Annotation | GPUs | Training Epochs | Joint Top1<br/>Config Link: Weight Link | Bone Top1<br/>Config Link: Weight Link | Joint Motion Top1<br/>Config Link: Weight Link | Bone-Motion Top1<br/>Config Link: Weight Link | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Vanilla | Official 3D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_vanilla_ntu60_xsub_3dkp/j.py): [81.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/stgcn/stgcn_vanilla_ntu60_xsub_3dkp/b.py): [81.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_vanilla_ntu60_xsub_3dkp/jm.py): [79.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_vanilla_ntu60_xsub_3dkp/bm.py): [81.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xsub_3dkp/bm.pth) | 84.3 | 86.6 |
| NTURGB+D XSub | Vanilla | HRNet 2D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_vanilla_ntu60_xsub_hrnet/j.py): [85.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xsub_hrnet/j.pth) | [bone_config](/configs/stgcn/stgcn_vanilla_ntu60_xsub_hrnet/b.py): [85.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xsub_hrnet/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_vanilla_ntu60_xsub_hrnet/jm.py): [81.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_vanilla_ntu60_xsub_hrnet/bm.py): [83.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xsub_hrnet/bm.pth) | 88.8 | 90.1 |
| NTURGB+D XSub | PYSKL | Official 3D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/j.py): [87.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/b.py): [88.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/jm.py): [85.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/bm.py): [86.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/bm.pth) | 90.0 | 90.7 |
| NTURGB+D XSub | PYSKL | HRNet 2D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j.py): [89.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/j.pth) | [bone_config](/configs/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/b.py): [91.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/jm.py): [86.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/bm.py): [87.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xsub_hrnet/bm.pth) | 92.0 | 92.4 |
| NTURGB+D XView | Vanilla | Official 3D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_vanilla_ntu60_xview_3dkp/j.py): [90.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xview_3dkp/j.pth) | [bone_config](/configs/stgcn/stgcn_vanilla_ntu60_xview_3dkp/b.py): [87.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xview_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_vanilla_ntu60_xview_3dkp/jm.py): [88.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_vanilla_ntu60_xview_3dkp/bm.py): [88.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xview_3dkp/bm.pth) | 91.4 | 93.2 |
| NTURGB+D XView | Vanilla | HRNet 2D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_vanilla_ntu60_xview_hrnet/j.py): [92.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xview_hrnet/j.pth) | [bone_config](/configs/stgcn/stgcn_vanilla_ntu60_xview_hrnet/b.py): [90.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xview_hrnet/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_vanilla_ntu60_xview_hrnet/jm.py): [92.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xview_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_vanilla_ntu60_xview_hrnet/bm.py): [86.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_vanilla_ntu60_xview_hrnet/bm.pth) | 93.8 | 95.1 |
| NTURGB+D XView | PYSKL | Official 3D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_pyskl_ntu60_xview_3dkp/j.py): [95.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xview_3dkp/j.pth) | [bone_config](/configs/stgcn/stgcn_pyskl_ntu60_xview_3dkp/b.py): [95.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xview_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_pyskl_ntu60_xview_3dkp/jm.py): [93.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_pyskl_ntu60_xview_3dkp/bm.py): [92.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xview_3dkp/bm.pth) | 96.2 | 96.5 |
| NTURGB+D XView | PYSKL | HRNet 2D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_pyskl_ntu60_xview_hrnet/j.py): [98.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xview_hrnet/j.pth) | [bone_config](/configs/stgcn/stgcn_pyskl_ntu60_xview_hrnet/b.py): [96.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xview_hrnet/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_pyskl_ntu60_xview_hrnet/jm.py): [95.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xview_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_pyskl_ntu60_xview_hrnet/bm.py): [95.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu60_xview_hrnet/bm.pth) | 98.2 | 98.3 |
| NTURGB+D 120 XSub | PYSKL | Official 3D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/j.py): [82.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/j.pth) | [bone_config](/configs/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/b.py): [83.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/jm.py): [80.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/bm.py): [80.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/bm.pth) | 85.6 | 86.2 |
| NTURGB+D 120 XSub | PYSKL | HRNet 2D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/j.py): [80.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/j.pth) | [bone_config](/configs/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/b.py): [83.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/jm.py): [78.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/bm.py): [79.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xsub_hrnet/bm.pth) | 84.0 | 84.7 |
| NTURGB+D 120 XSet | PYSKL | Official 3D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_pyskl_ntu120_xset_3dkp/j.py): [84.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xset_3dkp/j.pth) | [bone_config](/configs/stgcn/stgcn_pyskl_ntu120_xset_3dkp/b.py): [85.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xset_3dkp/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_pyskl_ntu120_xset_3dkp/jm.py): [82.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xset_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_pyskl_ntu120_xset_3dkp/bm.py): [83.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xset_3dkp/bm.pth) | 87.5 | 88.4 |
| NTURGB+D 120 XSet | PYSKL | HRNet 2D Skeleton | 8 | 80 | [joint_config](/configs/stgcn/stgcn_pyskl_ntu120_xset_hrnet/j.py): [84.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xset_hrnet/j.pth) | [bone_config](/configs/stgcn/stgcn_pyskl_ntu120_xset_hrnet/b.py): [87.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xset_hrnet/b.pth) | [joint_motion_config](/configs/stgcn/stgcn_pyskl_ntu120_xset_hrnet/jm.py): [82.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xset_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn/stgcn_pyskl_ntu120_xset_hrnet/bm.py): [83.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcn/stgcn_pyskl_ntu120_xset_hrnet/bm.pth) | 88.3 | 89.0 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train STGCN on NTURGB+D XSub (3D skeleton, Joint Modality) with 8 GPUs, with validation, with PYSKL practice, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/j.py 8 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test STGCN on NTURGB+D XSub (3D skeleton, Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/stgcn/stgcn_pyskl_ntu60_xsub_3dkp/j.py checkpoints/SOME_CHECKPOINT.pth 8 --eval top_k_accuracy --out result.pkl
```
