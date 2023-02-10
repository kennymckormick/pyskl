# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os.path as osp
from mmcv.utils import print_log

from ..smp import intop
from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class GestureDataset(BaseDataset):
    """Pose dataset for action recognition.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str): The dataset split used. Allowed choices are 'train', 'val', 'test', 'train+val'.
        valid_frames_thr (int): The threshold of valid frame numbers. Default: 0.
        **kwargs: Keyword arguments for 'BaseDataset'.
    """

    label_names = [
        'Doing other things', 'Drumming Fingers', 'No gesture',  # 0
        'Pulling Hand In', 'Pulling Two Fingers In', 'Pushing Hand Away', 'Pushing Two Fingers Away',  # 3
        'Rolling Hand Backward', 'Rolling Hand Forward', 'Shaking Hand',  # 7
        'Sliding Two Fingers Down', 'Sliding Two Fingers Left',  # 10
        'Sliding Two Fingers Right', 'Sliding Two Fingers Up',  # 12
        'Stop Sign', 'Swiping Down', 'Swiping Left', 'Swiping Right', 'Swiping Up',  # 14
        'Dislike', 'Like', 'Turning Hand Clockwise', 'Turning Hand Counterclockwise',  # 19
        'Zooming In With Full Hand', 'Zooming In With Two Fingers',  # 23
        'Zooming Out With Full Hand', 'Zooming Out With Two Fingers',  # 25
        'Call', 'Fist', 'Four', 'Mute', 'OK', 'One', 'Palm',  # 27
        'Peace', 'Rock', 'Three-Middle', 'Three-Left', 'Two Up', 'No Gesture'  # 34
    ]

    def __init__(self,
                 ann_file,
                 pipeline,
                 split,
                 valid_frames_thr=0,
                 squeeze=True,
                 mode='2D',
                 subset=None,
                 **kwargs):
        modality = 'Pose'
        self.split = split
        self.valid_frames_thr = valid_frames_thr
        self.squeeze = squeeze
        self.mode = mode
        self.subset = subset
        super().__init__(ann_file, pipeline, start_index=0, modality=modality, **kwargs)
        logger = get_root_logger()
        logger.info(f'{len(self)} videos remain after valid thresholding')

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        return self.load_pkl_annotations()

    def load_pkl_annotations(self):
        data = mmcv.load(self.ann_file)

        split, data = data['split'], data['annotations']

        identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
        if self.split == 'train+val':
            split = set(split['train'] + split['val'])
        else:
            split = set(split[self.split])

        data = [x for x in data if x[identifier] in split]
        if 'train' in self.split:
            if hasattr(data[0], 'valid_frames'):
                data = [x for x in data if x['valid_frames'] >= self.valid_frames_thr]

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])

            if len(item['keypoint'].shape) == 2:
                item['keypoint'] = item['keypoint'][None, None]
            elif self.squeeze and len(item['keypoint'].shape) == 4:
                keypoint = item['keypoint']
                assert keypoint.shape[0] == 1
                flag = (keypoint[0, ..., 2] > 0).sum(axis=1) > 0
                item['total_frames'] = flag.sum()
                item['keypoint'] = keypoint[:, flag]
                item['hand_score'] = item['hand_score'][:, flag]
                item['hand_lr'] = item['hand_lr'][:, flag]

            if self.mode == '2D':
                item['keypoint'] = item['keypoint'][..., :2]

            if self.subset is not None:
                data = [x for x in data if x['label'] in self.subset]

        return data

    def evaluate(self, results, logger=None, **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Returns:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        gt_labels = [ann['label'] for ann in self.video_infos]

        hit1 = intop(results, gt_labels, 1)
        hit5 = intop(results, gt_labels, 5)
        top1, top5 = np.mean(hit1), np.mean(hit5)

        eval_results = dict(top1_acc=top1, top5_acc=top5)
        log_msg = f'Top-1: {top1:.4f} Top-5: {top5:.4f}\n'
        log_msg += 'Per-Class Accuracy: \n'
        for i, label in enumerate(self.label_names):
            sub_hit = [h for h, gt in zip(hit1, gt_labels) if gt == i]
            if len(sub_hit):
                acc = np.mean(sub_hit)
                population = len(sub_hit)
                log_msg += f'Index: {i}, Action: {label}, Top-1: {acc}, Population: {population}\n'

        if 'total_frames' in self.video_infos[0]:
            log_msg += 'Average Accuracy of Videos with more than N skeletons: \n'
            valid_frames = [ann['valid_frames'] for ann in self.video_infos]
            numbers = [1, 5, 10, 15, 20]
            for n in numbers:
                sub_hit = [h for h, v in zip(hit1, valid_frames) if v >= n]
                acc = np.mean(sub_hit)
                population = len(sub_hit)

                log_msg += f'N: {n}, Top-1: {acc}, Population: {population}\n'

                if self.valid_frames_thr == n:
                    eval_results['top1_acc'] = acc
                    sub_hit5 = [h for h, v in zip(hit5, valid_frames) if v >= n]
                    eval_results['top5_acc'] = np.mean(sub_hit5)

        print_log(log_msg, logger=logger)
        return eval_results
