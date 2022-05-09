# CTRGCN

## Abstract

Graph convolutional networks (GCNs) have been widely used and achieved remarkable results in skeleton-based action recognition. In GCNs, graph topology dominates feature aggregation and therefore is the key to extracting representative features. In this work, we propose a novel Channel-wise Topology Refinement Graph Convolution (CTR-GC) to dynamically learn different topologies and effectively aggregate joint features in different channels for skeleton-based action recognition. The proposed CTR-GC models channel-wise topologies through learning a shared topology as a generic prior for all channels and refining it with channel-specific correlations for each channel. Our refinement method introduces few extra parameters and significantly reduces the difficulty of modeling channel-wise topologies. Furthermore, via reformulating graph convolutions into a unified form, we find that CTR-GC relaxes strict constraints of graph convolutions, leading to stronger representation capability. Combining CTR-GC with temporal modeling modules, we develop a powerful graph convolutional network named CTR-GCN which notably outperforms state-of-the-art methods on the NTU RGB+D, NTU RGB+D 120, and NW-UCLA datasets.

## Citation

```BibTeX
@inproceedings{chen2021channel,
  title={Channel-wise topology refinement graph convolution for skeleton-based action recognition},
  author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13359--13368},
  year={2021}
}
```

## Model Zoo

We release numerous checkpoints trained with various modalities, annotations on NTURGB+D and NTURGB+D 120. The accuracy of each modality links to the weight file.

| Dataset | Annotation | GPUs | Joint Top1 | Bone Top1 | Joint Motion Top1 | Bone-Motion Top1 | Two-Stream Top1 | Four Stream Top1 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| NTURGB+D XSub | Official 3D Skeleton | 8 | [joint_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/j.py): [89.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/j.pth) | [bone_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/b.py): [90.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/b.pth) | [joint_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/jm.py): [88.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/bm.py): [87.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/bm.pth) | 91.5 | 92.1 |
| NTURGB+D XSub | HRNet 2D Skeleton | 8 | [joint_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/j.py): [90.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/j.pth) | [bone_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/b.py): [92.7](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/b.pth) | [joint_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/jm.py): [89.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/bm.py): [90.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xsub_hrnet/bm.pth) | 93.3 | 93.6 |
| NTURGB+D XView | Official 3D Skeleton | 8 | [joint_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xview_3dkp/j.py): [95.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xview_3dkp/j.pth) | [bone_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xview_3dkp/b.py): [95.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xview_3dkp/b.pth) | [joint_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xview_3dkp/jm.py): [94.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xview_3dkp/jm.pth) | [bone_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xview_3dkp/bm.py): [93.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xview_3dkp/bm.pth) | 96.6 | 97.0 |
| NTURGB+D XView | HRNet 2D Skeleton | 8 | [joint_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xview_hrnet/j.py): [96.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xview_hrnet/j.pth) | [bone_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xview_hrnet/b.py): [97.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xview_hrnet/b.pth) | [joint_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xview_hrnet/jm.py): [94.8](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xview_hrnet/jm.pth) | [bone_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu60_xview_hrnet/bm.py): [95.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu60_xview_hrnet/bm.pth) | 98.4 | 98.4 |
| NTURGB+D 120 XSub | Official 3D Skeleton | 8 | [joint_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/j.py): [84.0](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/j.pth) | [bone_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/b.py): [85.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/b.pth) | [joint_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/jm.py): [81.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/jm.pth) | [bone_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/bm.py): [82.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_3dkp/bm.pth) | 87.5 | 88.1 |
| NTURGB+D 120 XSub | HRNet 2D Skeleton | 8 | [joint_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xsub_hrnet/j.py): [82.2](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_hrnet/j.pth) | [bone_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xsub_hrnet/b.py): [84.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_hrnet/b.pth) | [joint_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xsub_hrnet/jm.py): [82.3](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_hrnet/jm.pth) | [bone_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xsub_hrnet/bm.py): [82.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xsub_hrnet/bm.pth) | 85.8 | 86.6 |
| NTURGB+D 120 XSet | Official 3D Skeleton | 8 | [joint_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xset_3dkp/j.py): [85.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xset_3dkp/j.pth) | [bone_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xset_3dkp/b.py): [87.4](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xset_3dkp/b.pth) | [joint_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xset_3dkp/jm.py): [84.1](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xset_3dkp/jm.pth) | [bone_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xset_3dkp/bm.py): [83.9](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xset_3dkp/bm.pth) | 89.2 | 89.9 |
| NTURGB+D 120 XSet | HRNet 2D Skeleton | 8 | [joint_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xset_hrnet/j.py): [84.5](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xset_hrnet/j.pth) | [bone_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xset_hrnet/b.py): [88.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xset_hrnet/b.pth) | [joint_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xset_hrnet/jm.py): [85.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xset_hrnet/jm.pth) | [bone_motion_config](/configs/ctrgcn/ctrgcn_pyskl_ntu120_xset_hrnet/bm.py): [85.6](http://download.openmmlab.com/mmaction/pyskl/ckpt/ctrgcn/ctrgcn_pyskl_ntu120_xset_hrnet/bm.pth) | 89.0 | 90.1 |

**Note**

1. We use the linear-scaling learning rate (**Initial LR ∝ Batch Size**). If you change the training batch size, remember to change the initial LR proportionally.
2. For Two-Stream results, we adopt the **1 (Joint):1 (Bone)** fusion. For Four-Stream results, we adopt the **2 (Joint):2 (Bone):1 (Joint Motion):1 (Bone Motion)** fusion.


## Training & Testing

You can use the following command to train a model.

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train CTRGCN on NTURGB+D XSub (3D skeleton, Joint Modality) with 8 GPUs, with validation, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/j.py 8 --validate --test-last --test-best
```

You can use the following command to test a model.

```shell
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test CTRGCN on NTURGB+D XSub (3D skeleton, Joint Modality) with metrics `top_k_accuracy`, and dump the result to `result.pkl`.
bash tools/dist_test.sh configs/ctrgcn/ctrgcn_pyskl_ntu60_xsub_3dkp/j.py checkpoints/SOME_CHECKPOINT.pth 8 --eval top_k_accuracy --out result.pkl
```
