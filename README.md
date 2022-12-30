# PYSKL

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-skeleton-based-action-recognition/skeleton-based-action-recognition-on-ntu-rgbd)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd?p=revisiting-skeleton-based-action-recognition)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dg-stgcn-dynamic-spatial-temporal-modeling/skeleton-based-action-recognition-on-ntu-rgbd-1)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-ntu-rgbd-1?p=dg-stgcn-dynamic-spatial-temporal-modeling)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-skeleton-based-action-recognition/skeleton-based-action-recognition-on-kinetics)](https://paperswithcode.com/sota/skeleton-based-action-recognition-on-kinetics?p=revisiting-skeleton-based-action-recognition)

PYSKL is a toolbox focusing on action recognition based on **SK**e**L**eton data with **PY**Torch. Various algorithms will be supported for skeleton-based action recognition. We build this project based on the OpenSource Project [MMAction2](https://github.com/open-mmlab/mmaction2).

This repo is the official implementation of [PoseConv3D](https://arxiv.org/abs/2104.13586) and [STGCN++](https://github.com/kennymckormick/pyskl/tree/main/configs/stgcn%2B%2B).

<div align="center">
  <img src="https://user-images.githubusercontent.com/34324155/123989146-2ecae680-d9fb-11eb-916b-b9db5563a9e5.gif" width="500px"><br>
  <p style="font-size:1.2vw;">Skeleton-base Action Recognition Results on NTU-RGB+D-120</p>
</div>

## News

- Provide [scripts](/examples/inference_speed.ipynb) to estimate the inference speed of each model (**2022-12-30**).
- Support [RGBPoseConv3D](https://arxiv.org/abs/2104.13586), a two-stream 3D-CNN for action recognition based on RGB & Human Skeleton. Follow the [guide](/configs/rgbpose_conv3d/README.md) to train and test RGBPoseConv3D on NTURGB+D （**2022-12-29**).
- We provide a script ([ntu_preproc.py](/tools/data/ntu_preproc.py)) to generate PYSKL-style annotations files from official NTURGB+D skeleton files (**2022-12-20**).
- Support [DG-STGCN](https://arxiv.org/abs/2210.05895), which is a state-of-the-art skeleton action algorithm that doesn't rely on a pre-defined graph (**2022-12-12**).
- The [tech report](https://arxiv.org/abs/2205.09443) of PYSKL is accepted by MM 2022 (**2022-06-28**).

## Supported Algorithms

- [x] DG-STGCN (Arxiv): https://arxiv.org/abs/2210.05895 [[MODELZOO](/configs/dgstgcn/README.md)]
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

## Installation
```shell
git clone https://github.com/kennymckormick/pyskl.git
cd pyskl
# Please first install pytorch according to instructions on the official website: https://pytorch.org/get-started/locally/. Please use pytorch with version smaller than 1.11.0 and larger (or equal) than 1.5.0
# The following command will install mmcv-full 1.5.0 from source, which might be very slow (take ~10 minutes). You can also follow the instruction at https://github.com/open-mmlab/mmcv to install mmcv-full from pre-built wheels, which will be much faster.
pip install -r requirements.txt
pip install -e .
```

## Demo

```shell
# Before running the demo, make sure you have installed mmcv-full, mmpose and mmdet. You should first install mmcv-full, and then install mmpose, mmdet.
# You should run the following scripts under the directory `$PYSKL`
# Running the demo with PoseC3D trained on NTURGB+D 120 (Joint Modality), which is the default option. The input file is demo/ntu_sample.avi, the output file is demo/demo.mp4
python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo.mp4
# Running the demo with STGCN++ trained on NTURGB+D 120 (Joint Modality). The input file is demo/ntu_sample.avi, the output file is demo/demo.mp4
python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo.mp4 --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
```

Note that for running demo on an arbitrary input video, you need a tracker to formulate pose estimation results for each frame into multiple skeleton sequences. Currently we are using a [naive tracker](https://github.com/kennymckormick/pyskl/blob/4ddb7ac384e231694fd2b4b7774144e5762862ab/demo/demo_skeleton.py#L192) based on inter-frame pose similarities. You can also try to write your own tracker.

## Data Preparation

We provide HRNet 2D skeletons for every dataset we supported and Kinect 3D skeletons for the NTURGB+D and NTURGB+D 120 dataset. To obtain the human skeleton annotations, you can:

1. Use our pre-processed skeleton annotations: we directly provide the processed skeleton data for all datasets as pickle files (which can be directly used for training and testing), check [Data Doc](/tools/data/README.md) for the download links and descriptions of the annotation format.
2. For NTURGB+D 3D skeletons, you can download the official annotations from https://github.com/shahroudy/NTURGB-D, and use our [provided script](/tools/data/ntu_preproc.py) to generate the processed pickle files. The generated files are the same with the provided `ntu60_3danno.pkl` and `ntu120_3danno.pkl`. For detailed instructions, follow the [Data Doc](/tools/data/README.md).

3. We also provide scripts to extract 2D HRNet skeletons from RGB videos, you can follow the [diving48_example](/examples/extract_diving48_skeleton/diving48_example.ipynb) to extract 2D skeletons from an arbitrary RGB video dataset.

You can use [vis_skeleton](/demo/vis_skeleton.ipynb) to visualize the provided skeleton data.

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

## Contributing

PYSKL is an OpenSource Project under the Apache2 license. Any contribution from the community to improve PYSKL is appreciated. For **significant** contributions (like supporting a novel & important task), a corresponding part will be added to our updated tech report, while the contributor will also be added to the author list.

Any user can open a PR to contribute to PYSKL. The PR will be reviewed before being merged into the master branch. If you want to open a **large** PR in PYSKL, you are recommended to first reach me (via my email dhd.efz@gmail.com) to discuss the design, which helps to save large amounts of time in the reviewing stage.

## Contact

For any questions, feel free to contact: dhd.efz@gmail.com
