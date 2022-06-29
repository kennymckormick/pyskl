# PYSKL

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-skeleton-based-action-recognition/skeleton-based-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd?p=revisiting-skeleton-based-action-recognition) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pyskl-towards-good-practices-for-skeleton/skeleton-based-action-recognition-on-ntu-rgbd-1)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd-1?p=pyskl-towards-good-practices-for-skeleton) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-skeleton-based-action-recognition/skeleton-based-action-recognition-on-kinetics)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-kinetics?p=revisiting-skeleton-based-action-recognition)

PYSKL is a toolbox focusing on action recognition based on **SK**e**L**eton data with **PY**Torch. Various algorithms will be supported for skeleton-based action recognition. We build this project based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2).

This repo is the official implementation of [PoseConv3D](https://arxiv.org/abs/2104.13586) and [STGCN++](https://github.com/kennymckormick/pyskl/tree/main/configs/stgcn%2B%2B).

<div align="center">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="500px"><br>
  <p style="font-size:1.2vw;">Skeleton-base Action Recognition Results on NTU-RGB+D-120</p>
</div>

## Contributing

PYSKL is an OpenSource Project under the Apache2 license. Any contribution from the community to improve PYSKL is appreciated. For **significant** contributions (like supporting a novel & important task), a corresponding part will be added to our updated tech report, while the contributor will also be added to the author list.

Any user can open a PR to contribute to PYSKL. The PR will be reviewed before being merged into the master branch. If you want to open a **large** PR in PYSKL, you are recommended to first reach me (via my email dhd.efz@gmail.com) to discuss the design, which helps to save large amounts of time in the reviewing stage.

## News

- The [tech report](https://arxiv.org/abs/2205.09443) of PYSKL is accepted by MM 2022 (**2022-06-28**).
- Support spatial augmentations and provide a benchmark on ST-GCN++  (**2022-05-12**).
- Support skeleton action recognition demo with GCN algorithms  (**2022-05-03**).
- Release the skeleton annotations (HRNet 2D Pose), config files, and pre-trained ckpts for Kinetics-400. K400 is a large-scale dataset (even for skeleton), you should have `memcached` and `pymemcache` installed for efficient training & testing on K400 (**2022-05-01**).
- Provide an example (diving48) for processing a custom video dataset, generating 2D skeleton annotations, and using PoseC3D for skeleton-based action recognition. The tutorial for skeleton extraction part is available in [diving48_example](/examples/extract_diving48_skeleton/diving48_example.ipynb)  (**2022-04-15**).

## Supported Algorithms

- [x] ST-GCN (AAAI 2018): https://arxiv.org/abs/1801.07455 [[MODELZOO](/configs/stgcn/README.md)]
- [x] ST-GCN++ (PYSKL, Tech Report): https://arxiv.org/abs/2205.09443 [[MODELZOO](/configs/stgcn++/README.md)]
- [x] PoseConv3D (CVPR 2022 Oral): https://arxiv.org/abs/2104.13586 [[MODELZOO](/configs/posec3d/README.md)]
- [x] AAGCN (TIP): https://arxiv.org/abs/1912.06971 [[MODELZOO](/configs/aagcn/README.md)]
- [x] MS-G3D (CVPR 2020 Oral): https://arxiv.org/abs/2003.14111 [[MODELZOO](/configs/msg3d/README.md)]
- [x] CTR-GCN (ICCV 2021): https://arxiv.org/abs/2107.12213 [[MODELZOO](/configs/ctrgcn/README.md)]

## Supported Skeleton Datasets

- [x] NTURGB+D (CVPR 2016): [NTU RGB+D: A large scale dataset for 3D human activity analysis](https://openaccess.thecvf.com/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf)
- [x] NTURGB+D 120 (TPAMI 2019): [Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8713892)
- [x] Kinetics 400 (CVPR 2017): [Quo vadis, action recognition? a new model and the kinetics dataset](https://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf)
- [x] UCF101 (ArXiv 2012): [UCF101: A dataset of 101 human actions classes from videos in the wild](https://arxiv.org/pdf/1212.0402.pdf)
- [x] HMDB51 (ICCV 2021): [HMDB: a large video database for human motion recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6126543)
- [x] FineGYM (CVPR 2020): [Finegym: A hierarchical video dataset for fine-grained action understanding](https://openaccess.thecvf.com/content_CVPR_2020/papers/Shao_FineGym_A_Hierarchical_Video_Dataset_for_Fine-Grained_Action_Understanding_CVPR_2020_paper.pdf)
- [x] Diving48 (ECCV 2018): [Resound: Towards action recognition without representation bias](https://openaccess.thecvf.com/content_ECCV_2018/papers/Yingwei_Li_RESOUND_Towards_Action_ECCV_2018_paper.pdf)

For data pre-processing, we estimate 2D skeletons with a two-stage pose estimator (Faster-RCNN + HRNet). For 3D skeletons, we follow the pre-processing procedure of [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). Currently, we do not provide the pre-processing scripts. Instead, we directly provide the processed skeleton data as pickle files ([download links here](/tools/data/data_doc.md)), which can be directly used in training and evaluation.  You can use [vis_skeleton](/demo/vis_skeleton.ipynb) to visualize the provided skeleton data.

## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# Please first install pytorch according to instructions on the official website: https://pytorch.org/get-started/locally/. Please use pytorch with version smaller than 1.11.0 and larger (or equal) than 1.5.0
pip install -r requirements.txt
pip install -e .
```

## Demonstration

```shell
# Before running the demo, make sure you have installed mmcv-full, mmpose and mmdet. You should first install mmcv-full, and then install mmpose, mmdet.
# You should run the following scripts under the directory `$PYSKL`
# Running the demo with PoseC3D trained on NTURGB+D 120 (Joint Modality), which is the default option. The input file is demo/ntu_sample.avi, the output file is demo/demo.mp4
python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo.mp4
# Running the demo with STGCN++ trained on NTURGB+D 120 (Joint Modality). The input file is demo/ntu_sample.avi, the output file is demo/demo.mp4
python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo.mp4 --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
```

Note that for running demo on an arbitrary input video, you need a tracker to formulate pose estimation results for each frame into multiple skeleton sequences. Currently we are using a [naive tracker](https://github.com/kennymckormick/pyskl/blob/4ddb7ac384e231694fd2b4b7774144e5762862ab/demo/demo_skeleton.py#L192) based on inter-frame pose similarities. You can also try to write your own tracker.

## Training & Testing

You can use following commands for training and testing. Basically, we support distributed training on a single server with multiple GPUs.
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
@misc{duan2022PYSKL,
  url = {https://arxiv.org/abs/2205.09443},
  author = {Duan, Haodong and Wang, Jiaqi and Chen, Kai and Lin, Dahua},
  title = {PYSKL: Towards Good Practices for Skeleton Action Recognition},
  publisher = {arXiv},
  year = {2022}
}
```

## Contact
For any questions, feel free to contact: dh019@ie.cuhk.edu.hk
