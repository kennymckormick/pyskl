# Demo

We currently provide an offline GPU-demo for skeleton action recognition and an online CPU-demo for gesture recognition. Details are provided below.

## Preparation

- Before running the skeleton action recognition demo, make sure you have installed `mmcv-full`, `mmpose` and `mmdet`. We recommend you to directly use the provided conda environment, with all necessary dependencies included:
```bash
# Following commands assume you are in the root directory of pyskl (indicated as `$PYSKL`)
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml  # Create the conda environment (named `pyskl`) for this project, run it if you haven't created one yet.
conda activate pyskl  # Activate the `pyskl` environment
pip install -e .  # Install this project
```
- Before running the gesture recognition demo, you need to install `mediapipe` first. This can be completed simply by `pip install mediapipe`.

## Skeleton Action Recognition Demo (GPU, offline)

The provided skeleton action recognition demo is offline, which means it takes a video clip as input and return the action detection. The demo runs on GPU. By default, this demo recognizes 120 actions categories defined in [NTURGB+D 120](https://arxiv.org/abs/1905.04757).

For human skeleton extraction, we use [Faster-RCNN (R50 backbone)](/demo/faster_rcnn_r50_fpn_2x_coco.py) for human detection and [HRNet_w32](demo/hrnet_w32_coco_256x192.py) for human pose estimation. All based on OpenMMLab implementations.

```bash
# Running the demo with PoseC3D trained on NTURGB+D 120 (Joint Modality), which is the default option. The input file is demo/ntu_sample.avi, the output file is demo/demo.mp4
python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo.mp4
# Running the demo with STGCN++ trained on NTURGB+D 120 (Joint Modality). The input file is demo/ntu_sample.avi, the output file is demo/demo.mp4
python demo/demo_skeleton.py demo/ntu_sample.avi demo/demo.mp4 --config configs/stgcn++/stgcn++_ntu120_xsub_hrnet/j.py --checkpoint http://download.openmmlab.com/mmaction/pyskl/ckpt/stgcnpp/stgcnpp_ntu120_xsub_hrnet/j.pth
```

Note that for running demo on an arbitrary input video, you need a tracker to formulate pose estimation results for each frame into multiple skeleton sequences. Currently we are using a [naive tracker](https://github.com/kennymckormick/pyskl/blob/4ddb7ac384e231694fd2b4b7774144e5762862ab/demo/demo_skeleton.py#L192) based on inter-frame pose similarities. You can also try to write your own tracker.

## Gestrue Recognition Demo (CPU, Real-time)

We provide an online gesture recognition demo that runs real-time on CPU. The demo takes a video stream as input and predict the current gesture performed (It only supports the single-hand scenario now). By default, this demo recognizes 15 gestures defined in [HaGRID](https://github.com/hukenovs/hagrid), including: Call, Dislike, Fist, Four, Like, Mute, OK, One, Palm, Peace, Rock, Stop, Three [Middle 3 Fingers], Three [Left 3 Fingers], Two Up.

For hand keypoint extraction, we use the opensource solution [mediapipe](https://google.github.io/mediapipe/). For skeleton-based gesture recognition, currently we adopt a light variant of [ST-GCN++](/demo/stgcnpp_gesture.py) model trained on the [HaGRID](https://github.com/hukenovs/hagrid) gesture recognition dataset.

```bash
# Run the real time skeleton-based gesture recognition demo
python demo/demo_gesture.py
```
