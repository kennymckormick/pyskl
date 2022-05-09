# AAGCN

## Abstract

Graph convolutional networks (GCNs), which generalize CNNs to more generic non-Euclidean structures, have achieved remarkable performance for skeleton-based action recognition. However, there still exist several issues in the previous GCN-based models. First, the topology of the graph is set heuristically and fixed over all the model layers and input data. This may not be suitable for the hierarchy of the GCN model and the diversity of the data in action recognition tasks. Second, the second-order information of the skeleton data, i.e., the length and orientation of the bones, is rarely investigated, which is naturally more informative and discriminative for the human action recognition. In this work, we propose a novel multi-stream attention-enhanced adaptive graph convolutional neural network (MS-AAGCN) for skeleton-based action recognition. The graph topology in our model can be either uniformly or individually learned based on the input data in an end-to-end manner. This data-driven approach increases the flexibility of the model for graph construction and brings more generality to adapt to various data samples. Besides, the proposed adaptive graph convolutional layer is further enhanced by a spatial-temporal-channel attention module, which helps the model pay more attention to important joints, frames and features. Moreover, the information of both the joints and bones, together with their motion information, are simultaneously modeled in a multi-stream framework, which shows notable improvement for the recognition accuracy. Extensive experiments on the two large-scale datasets, NTU-RGBD and Kinetics-Skeleton, demonstrate that the performance of our model exceeds the state-of-the-art with a significant margin.

## Citation

```BibTeX
@article{shi2020skeleton,
  title={Skeleton-based action recognition with multi-stream adaptive graph convolutional networks},
  author={Shi, Lei and Zhang, Yifan and Cheng, Jian and Lu, Hanqing},
  journal={IEEE Transactions on Image Processing},
  volume={29},
  pages={9532--9545},
  year={2020},
  publisher={IEEE}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | GPUs | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone-Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | 8 | [joint_config](/configs/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/j.py): [89.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/b.py): [89.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/jm.py): [86.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/bm.py): [86.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/bm.pth) | 90.8 | 91.5 |
| NTURGB+D XSub | HRNet 2D Skeleton | 8 | [joint_config](/configs/aagcn/aagcn_pyskl_ntu60_xsub_hrnet/j.py): [89.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xsub_hrnet/j.pth) | [bone_config](/configs/aagcn/aagcn_pyskl_ntu60_xsub_hrnet/b.py): [92.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xsub_hrnet/b.pth) | [joint_motion_config](/configs/aagcn/aagcn_pyskl_ntu60_xsub_hrnet/jm.py): [88.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/aagcn/aagcn_pyskl_ntu60_xsub_hrnet/bm.py): [88.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xsub_hrnet/bm.pth) | 92.8 | 93.0 |
| NTURGB+D XView | Official 3D Skeleton | 8 | [joint_config](/configs/aagcn/aagcn_pyskl_ntu60_xview_3dkp/j.py): [95.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xview_3dkp/j.pth) | [bone_config](/configs/aagcn/aagcn_pyskl_ntu60_xview_3dkp/b.py): [95.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xview_3dkp/b.pth) | [joint_motion_config](/configs/aagcn/aagcn_pyskl_ntu60_xview_3dkp/jm.py): [93.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/aagcn/aagcn_pyskl_ntu60_xview_3dkp/bm.py): [92.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xview_3dkp/bm.pth) | 96.4 | 96.7 |
| NTURGB+D XView | HRNet 2D Skeleton | 8 | [joint_config](/configs/aagcn/aagcn_pyskl_ntu60_xview_hrnet/j.py): [97.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xview_hrnet/j.pth) | [bone_config](/configs/aagcn/aagcn_pyskl_ntu60_xview_hrnet/b.py): [96.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xview_hrnet/b.pth) | [joint_motion_config](/configs/aagcn/aagcn_pyskl_ntu60_xview_hrnet/jm.py): [95.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xview_hrnet/jm.pth) | [bone_motion_config](/configs/aagcn/aagcn_pyskl_ntu60_xview_hrnet/bm.py): [95.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu60_xview_hrnet/bm.pth) | 97.8 | 98.2 |
| NTURGB+D 120 XSub | Official 3D Skeleton | 8 | [joint_config](/configs/aagcn/aagcn_pyskl_ntu120_xsub_3dkp/j.py): [82.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xsub_3dkp/j.pth) | [bone_config](/configs/aagcn/aagcn_pyskl_ntu120_xsub_3dkp/b.py): [84.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xsub_3dkp/b.pth) | [joint_motion_config](/configs/aagcn/aagcn_pyskl_ntu120_xsub_3dkp/jm.py): [80.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/aagcn/aagcn_pyskl_ntu120_xsub_3dkp/bm.py): [80.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xsub_3dkp/bm.pth) | 86.3 | 86.9 |
| NTURGB+D 120 XSub | HRNet 2D Skeleton | 8 | [joint_config](/configs/aagcn/aagcn_pyskl_ntu120_xsub_hrnet/j.py): [80.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xsub_hrnet/j.pth) | [bone_config](/configs/aagcn/aagcn_pyskl_ntu120_xsub_hrnet/b.py): [84.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xsub_hrnet/b.pth) | [joint_motion_config](/configs/aagcn/aagcn_pyskl_ntu120_xsub_hrnet/jm.py): [80.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/aagcn/aagcn_pyskl_ntu120_xsub_hrnet/bm.py): [81.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xsub_hrnet/bm.pth) | 84.7 | 85.5 |
| NTURGB+D 120 XSet | Official 3D Skeleton | 8 | [joint_config](/configs/aagcn/aagcn_pyskl_ntu120_xset_3dkp/j.py): [84.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xset_3dkp/j.pth) | [bone_config](/configs/aagcn/aagcn_pyskl_ntu120_xset_3dkp/b.py): [86.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xset_3dkp/b.pth) | [joint_motion_config](/configs/aagcn/aagcn_pyskl_ntu120_xset_3dkp/jm.py): [82.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xset_3dkp/jm.pth) | [bone_motion_config](/configs/aagcn/aagcn_pyskl_ntu120_xset_3dkp/bm.py): [82.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xset_3dkp/bm.pth) | 88.1 | 88.8 |
| NTURGB+D 120 XSet | HRNet 2D Skeleton | 8 | [joint_config](/configs/aagcn/aagcn_pyskl_ntu120_xset_hrnet/j.py): [86.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xset_hrnet/j.pth) | [bone_config](/configs/aagcn/aagcn_pyskl_ntu120_xset_hrnet/b.py): [88.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xset_hrnet/b.pth) | [joint_motion_config](/configs/aagcn/aagcn_pyskl_ntu120_xset_hrnet/jm.py): [85.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xset_hrnet/jm.pth) | [bone_motion_config](/configs/aagcn/aagcn_pyskl_ntu120_xset_hrnet/bm.py): [85.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/aagcn/aagcn_pyskl_ntu120_xset_hrnet/bm.pth) | 89.1 | 89.9 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train AAGCN on NTURGB+D XSub (3D skeleton, Joint Modality) with 8 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/j.py 8 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test AAGCN on NTURGB+D XSub (3D skeleton, Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/aagcn/aagcn_pyskl_ntu60_xsub_3dkp/j.py checkpoints/SOME_CHECKPOINT.pth 8 --eval top_k_accuracy --out result.pkl
```
