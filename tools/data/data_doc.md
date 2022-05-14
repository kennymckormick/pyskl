# Things you need to know about PYSKL data format

PYSKL now provides pre-processed pickle annotations files for training and testing. The pre-processing scripts will be released in later updates. Below we demonstrate the format of the annotation files and provide the download links.

## The format of the pickle files

Each pickle file corresponds to an action recognition dataset. The content of a pickle file is a dictionary with two fields: `split` and `annotations`

1. Split: The value of the `split` field is a dictionary: the keys are the split names, while the values are lists of video identifiers that belong to the specific clip.
2. Annotations: The value of the `annotations` field is a list of skeleton annotations, each skeleton annotation is a dictionary, containing the following fields:
   1. `frame_dir` (str): The identifier of the corresponding video.
   2. `total_frames` (int): The number of frames in this video.
   3. `img_shape` (tuple[int]): The shape of a video frame, a tuple with two elements, in the format of (height, width). Only required for 2D skeletons.
   4. `original_shape` (tuple[int]): Same as `img_shape`.
   5. `label` (int): The action label.
   6. `keypoint` (np.ndarray, with shape [M x T x V x C]): The keypoint annotation. M: number of persons; T: number of frames (same as `total_frames`); V: number of keypoints (25 for NTURGB+D 3D skeleton, 17 for CoCo, 18 for OpenPose, etc. ); C: number of dimensions for keypoint coordinates (C=2 for 2D keypoint, C=3 for 3D keypoint).
   7. `keypoint_score` (np.ndarray, with shape [M x T x V]): The confidence score of keypoints. Only required for 2D skeletons.

Note:
1. For Kinetics400, things are a little different (for storage saving and training acceleration):
   1. The fields `keypoint`, `keypoint_score` are not in the annotation file, but stored in many different **kpfiles**.
   2. A new field named `raw_file`, which specifies the file path of the **kpfile** that contains the skeleton annotation of this video.
   3. Each **kpfile** is a dictionary: key is the `frame_dir`, value is a dictionary with a single key `keypoint`. The value of `keypoint` is an ndarray with shape [N x V x C]. N: number of skeletons in the video; V: number of keypoints; C (C=3): number of dimensions for keypoint (x, y, score).
   4. A new field named `frame_inds`, indicates the corresponding frame index of each skeleton.
   5. A new field named `box_score`, indicates the corresponding bbox score of each skeleton.
   6. A new field named `valid`, indicates how many frames (with valid skeletons) left when we only keep skeletons with bbox scores larger than a threshold.
   7. We cache the kpfiles in memory with memcache and query with `frame_dir` to obtain the skeleton annotation. Kinetics-400 skeletons are converted to normal skeleton format with operator `DecompressPose`.

You can download an annotation file and browse it to get familiar with our annotation formats.

## Download the pre-processed skeletons

We provide links to the pre-processed skeleton annotations, you can directly download them and use them for training & testing.

- NTURGB+D [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_hrnet.pkl
- NTURGB+D [3D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu60_3danno.pkl
- NTURGB+D 120 [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_hrnet.pkl
- NTURGB+D 120 [3D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/nturgbd/ntu120_3danno.pkl
- GYM [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/gym/gym_hrnet.pkl
- UCF101 [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/ucf101/ucf101_hrnet.pkl
- HMDB51 [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/hmdb51/hmdb51_hrnet.pkl
- Diving48 [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/diving48/diving48_hrnet.pkl
- Kinetics400 [2D Skeleton]: https://download.openmmlab.com/mmaction/pyskl/data/k400/k400_hrnet.pkl (Table of contents only, no skeleton annotations)

For Kinetics400, since the skeleton annotations are large, we do not provide the direct download links on aliyun. Please use the following link to download the `kpfiles` and extract it under `$PYSKL/data/k400` for Kinetics-400 training & testing: https://mycuhk-my.sharepoint.com/:u:/g/personal/1155136485_link_cuhk_edu_hk/EeyDCVskqLtClMVVwqD53acBF2FEwkctp3vtRbkLfnKSTw?e=B3SZlM

Here are the BibTex items for each dataset:

```BibTex
% NTURGB+D
@inproceedings{shahroudy2016ntu,
  title={Ntu rgb+ d: A large scale dataset for 3d human activity analysis},
  author={Shahroudy, Amir and Liu, Jun and Ng, Tian-Tsong and Wang, Gang},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={1010--1019},
  year={2016}
}
% NTURGB+D 120
@article{liu2019ntu,
  title={Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding},
  author={Liu, Jun and Shahroudy, Amir and Perez, Mauricio and Wang, Gang and Duan, Ling-Yu and Kot, Alex C},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={42},
  number={10},
  pages={2684--2701},
  year={2019},
  publisher={IEEE}
}
% Kinetics-400
@inproceedings{carreira2017quo,
  title={Quo vadis, action recognition? a new model and the kinetics dataset},
  author={Carreira, Joao and Zisserman, Andrew},
  booktitle={proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={6299--6308},
  year={2017}
}
% GYM
@inproceedings{shao2020finegym,
  title={Finegym: A hierarchical video dataset for fine-grained action understanding},
  author={Shao, Dian and Zhao, Yue and Dai, Bo and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={2616--2625},
  year={2020}
}
% UCF101
@article{soomro2012ucf101,
  title={UCF101: A dataset of 101 human actions classes from videos in the wild},
  author={Soomro, Khurram and Zamir, Amir Roshan and Shah, Mubarak},
  journal={arXiv preprint arXiv:1212.0402},
  year={2012}
}
% HMDB51
@inproceedings{kuehne2011hmdb,
  title={HMDB: a large video database for human motion recognition},
  author={Kuehne, Hildegard and Jhuang, Hueihan and Garrote, Est{\'\i}baliz and Poggio, Tomaso and Serre, Thomas},
  booktitle={2011 International conference on computer vision},
  pages={2556--2563},
  year={2011},
  organization={IEEE}
}
% Diving48
@inproceedings{li2018resound,
  title={Resound: Towards action recognition without representation bias},
  author={Li, Yingwei and Li, Yi and Vasconcelos, Nuno},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={513--528},
  year={2018}
}
```
