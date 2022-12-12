# DG-STGCN

## Abstract

Graph convolution networks (GCN) have been widely used in skeleton-based action recognition. We note that existing GCN-based approaches primarily rely on prescribed graphical structures (i.e., a manually defined topology of skeleton joints), which limits their flexibility to capture complicated correlations between joints. To move beyond this limitation, we propose a new framework for skeleton-based action recognition, namely Dynamic Group Spatio-Temporal GCN (DG-STGCN). It consists of two modules, DG-GCN and DG-TCN, respectively, for spatial and temporal modeling. In particular, DG-GCN uses learned affinity matrices to capture dynamic graphical structures instead of relying on a prescribed one, while DG-TCN performs group-wise temporal convolutions with varying receptive fields and incorporates a dynamic joint-skeleton fusion module for adaptive multi-level temporal modeling. On a wide range of benchmarks, including NTURGB+D, Kinetics-Skeleton, BABEL, and Toyota SmartHome, DG-STGCN consistently outperforms state-of-the-art methods, often by a notable margin.

## Citation

```BibTeX
@article{duan2022dg,
  title={DG-STGCN: Dynamic Spatial-Temporal Modeling for Skeleton-based Action Recognition},
  author={Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  journal={arXiv preprint arXiv:2210.05895},
  year={2022}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | GPUs | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone-Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | 8 | [joint_config](/configs/dgstgcn/ntu60_xsub_3dkp/j.py): [91.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/dgstgcn/ntu60_xsub_3dkp/b.py): [91.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/dgstgcn/ntu60_xsub_3dkp/jm.py): [88.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/dgstgcn/ntu60_xsub_3dkp/bm.py): [88.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xsub_3dkp/bm.pth) | 92.9 | 93.2 |
| NTURGB+D XView | Official 3D Skeleton | 8 | [joint_config](/configs/dgstgcn/ntu60_xview_3dkp/j.py): [96.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xview_3dkp/j.pth) | [bone_config](/configs/dgstgcn/ntu60_xview_3dkp/b.py): [96.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xview_3dkp/b.pth) | [joint_motion_config](/configs/dgstgcn/ntu60_xview_3dkp/jm.py): [95.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/dgstgcn/ntu60_xview_3dkp/bm.py): [94.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu60_xview_3dkp/bm.pth) | 97.4 | 97.5 |
| NTURGB+D 120 XSub | Official 3D Skeleton | 8 | [joint_config](/configs/dgstgcn/ntu120_xsub_3dkp/j.py): [85.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xsub_3dkp/j.pth) | [bone_config](/configs/dgstgcn/ntu120_xsub_3dkp/b.py): [88.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xsub_3dkp/b.pth) | [joint_motion_config](/configs/dgstgcn/ntu120_xsub_3dkp/jm.py): [82.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/dgstgcn/ntu120_xsub_3dkp/bm.py): [83.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xsub_3dkp/bm.pth) | 89.3 | 89.6 |
| NTURGB+D 120 XSet | Official 3D Skeleton | 8 | [joint_config](/configs/dgstgcn/ntu120_xset_3dkp/j.py): [87.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xset_3dkp/j.pth) | [bone_config](/configs/dgstgcn/ntu120_xset_3dkp/b.py): [89.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xset_3dkp/b.pth) | [joint_motion_config](/configs/dgstgcn/ntu120_xset_3dkp/jm.py): [85.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xset_3dkp/jm.pth) | [bone_motion_config](/configs/dgstgcn/ntu120_xset_3dkp/bm.py): [85.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/dgstgcn/ntu120_xset_3dkp/bm.pth) | 91.2 | 91.3 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train DG-STGCN on NTURGB+D XSub (3D skeleton, Joint Modality) with 8 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/dgstgcn/ntu60_xsub_3dkp/j.py 8 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test DG-STGCN on NTURGB+D XSub (3D skeleton, Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/dgstgcn/ntu60_xsub_3dkp/j.py checkpoints/SOME_CHECKPOINT.pth 8 --eval top_k_accuracy --out result.pkl
```
