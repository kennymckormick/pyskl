# STGCN++

## Introduction

STGCN++ is a variant of STGCN we developed in PYSKL with some modifications in the architecture of the spatial module and the temporal module. We provide STGCN++ trained on NTURGB+D with 2D skeletons (HRNet) and 3D skeletons with **PYSKL** training setting. We provide checkpoints for four modalities: Joint, Bone, Joint Motion, and Bone Motion. The architecture of STGCN++ is described in PYSKL [tech report](https://arxiv.org/abs/2205.09443).

## Citation

```BibTeX
@misc{duan2022PYSKL,
  url = {https://arxiv.org/abs/2205.09443},
  author = {Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  title = {PYSKL: Towards Good Practices for Skeleton Action Recognition},
  publisher = {arXiv},
  year = {2022}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | GPUs | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone-Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | 8 | [joint_config](/configs/stgcn++/stgcn++_ntu60_xsub_3dkp/j.py): [89.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/stgcn++/stgcn++_ntu60_xsub_3dkp/b.py): [90.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/stgcn++/stgcn++_ntu60_xsub_3dkp/jm.py): [87.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn++/stgcn++_ntu60_xsub_3dkp/bm.py): [87.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_3dkp/bm.pth) | 91.4 | 92.1 |
| NTURGB+D XSub | HRNet 2D Skeleton | 8 | [joint_config](/configs/stgcn++/stgcn++_ntu60_xsub_hrnet/j.py): [89.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/j.pth) | [bone_config](/configs/stgcn++/stgcn++_ntu60_xsub_hrnet/b.py): [92.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/b.pth) | [joint_motion_config](/configs/stgcn++/stgcn++_ntu60_xsub_hrnet/jm.py): [84.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn++/stgcn++_ntu60_xsub_hrnet/bm.py): [88.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xsub_hrnet/bm.pth) | 92.8 | 93.2 |
| NTURGB+D XView | Official 3D Skeleton | 8 | [joint_config](/configs/stgcn++/stgcn++_ntu60_xview_3dkp/j.py): [95.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/j.pth) | [bone_config](/configs/stgcn++/stgcn++_ntu60_xview_3dkp/b.py): [95.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/b.pth) | [joint_motion_config](/configs/stgcn++/stgcn++_ntu60_xview_3dkp/jm.py): [94.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn++/stgcn++_ntu60_xview_3dkp/bm.py): [93.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_3dkp/bm.pth) | 96.7 | 97.0 |
| NTURGB+D XView | HRNet 2D Skeleton | 8 | [joint_config](/configs/stgcn++/stgcn++_ntu60_xview_hrnet/j.py): [97.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_hrnet/j.pth) | [bone_config](/configs/stgcn++/stgcn++_ntu60_xview_hrnet/b.py): [97.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_hrnet/b.pth) | [joint_motion_config](/configs/stgcn++/stgcn++_ntu60_xview_hrnet/jm.py): [93.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn++/stgcn++_ntu60_xview_hrnet/bm.py): [95.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu60_xview_hrnet/bm.pth) | 98.4 | 98.5 |
| NTURGB+D 120 XSub | Official 3D Skeleton | 8 | [joint_config](/configs/stgcn++/stgcn++_ntu120_xsub_3dkp/j.py): [83.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/j.pth) | [bone_config](/configs/stgcn++/stgcn++_ntu120_xsub_3dkp/b.py): [85.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/b.pth) | [joint_motion_config](/configs/stgcn++/stgcn++_ntu120_xsub_3dkp/jm.py): [80.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn++/stgcn++_ntu120_xsub_3dkp/bm.py): [81.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_3dkp/bm.pth) | 87.0 | 87.5 |
| NTURGB+D 120 XSub | HRNet 2D Skeleton | 8 | [joint_config](/configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py): [84.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth) | [bone_config](/configs/stgcn++/stgcn++_ntu120_xsub_hrnet/b.py): [84.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/b.pth) | [joint_motion_config](/configs/stgcn++/stgcn++_ntu120_xsub_hrnet/jm.py): [76.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn++/stgcn++_ntu120_xsub_hrnet/bm.py): [81.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/bm.pth) | 86.4 | 86.4 |
| NTURGB+D 120 XSet | Official 3D Skeleton | 8 | [joint_config](/configs/stgcn++/stgcn++_ntu120_xset_3dkp/j.py): [85.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_3dkp/j.pth) | [bone_config](/configs/stgcn++/stgcn++_ntu120_xset_3dkp/b.py): [87.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_3dkp/b.pth) | [joint_motion_config](/configs/stgcn++/stgcn++_ntu120_xset_3dkp/jm.py): [84.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_3dkp/jm.pth) | [bone_motion_config](/configs/stgcn++/stgcn++_ntu120_xset_3dkp/bm.py): [83.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_3dkp/bm.pth) | 89.1 | 89.8 |
| NTURGB+D 120 XSet | HRNet 2D Skeleton | 8 | [joint_config](/configs/stgcn++/stgcn++_ntu120_xset_hrnet/j.py): [88.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_hrnet/j.pth) | [bone_config](/configs/stgcn++/stgcn++_ntu120_xset_hrnet/b.py): [88.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_hrnet/b.pth) | [joint_motion_config](/configs/stgcn++/stgcn++_ntu120_xset_hrnet/jm.py): [82.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_hrnet/jm.pth) | [bone_motion_config](/configs/stgcn++/stgcn++_ntu120_xset_hrnet/bm.py): [84.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xset_hrnet/bm.pth) | 90.0 | 90.3 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train STGCN++ on NTURGB+D XSub (3D skeleton, Joint Modality) with 8 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/stgcn++/stgcn++_ntu60_xsub_3dkp/j.py 8 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test STGCN++ on NTURGB+D XSub (3D skeleton, Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/stgcn++/stgcn++_ntu60_xsub_3dkp/j.py checkpoints/SOME_CHECKPOINT.pth 8 --eval top_k_accuracy --out result.pkl
```
