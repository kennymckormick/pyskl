# Copyright (c) OpenMMLab. All rights reserved.
import copy as cp
import numpy as np
import os
import os.path as osp
from collections import defaultdict
from datetime import datetime

from ..core import ava_map
from ..smp import mdump, mload
from ..utils import get_root_logger
from .base import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class AVAPoseDataset(BaseDataset):
    """Pose dataset for AVA spatio-temporal action detection.

    The dataset loads pose and apply specified transforms to return a dict containing pose information.

    Args:
        ann_file (str): Path to the annotation file.
        pipeline (list[dict | callable]): A sequence of data transforms.
        split (str | None): The dataset split used. Choices are 'train', 'val'. Default: None.
        box_thr (float): The threshold for human proposals. Only boxes with confidence score larger than `box_thr` is
            kept. Accepted value range is [0, 1]. Default: 0.
        memcached (bool): Whether keypoint is cached in memcached. If set as True, will use 'frame_dir' as the key to
            fetch 'keypoint' from memcached. Default: False.
        mc_cfg (tuple): The config for memcached client, only applicable if `memcached==True`.
            Default: ('localhost', 22077).
        **kwargs: Keyword arguments for 'BaseDataset'.
    """

    def __init__(self,
                 ann_file,
                 pipeline,
                 split=None,
                 box_thr=0,
                 memcached=False,
                 clip_length=10,
                 squeeze=False,
                 reweight=None,
                 mc_cfg=('localhost', 22077),
                 **kwargs):

        self.split = split
        multi_class = kwargs.pop('multi_class', True)
        num_classes = kwargs.pop('num_classes', 81)
        super().__init__(
            ann_file,
            pipeline,
            modality='Pose',
            multi_class=multi_class,
            num_classes=num_classes,
            memcached=memcached,
            mc_cfg=mc_cfg,
            **kwargs)

        self.box_thr = box_thr
        assert isinstance(self.box_thr, float)
        self.clip_length = clip_length
        self.squeeze = squeeze
        self.reweight = None if reweight is None else mload(reweight)

        # Thresholding Training Examples
        for item in self.video_infos:
            annos = item['data']
            if self.box_thr is not None and hasattr(annos[0][0], 'box_score'):
                annos = [[y for y in box_list if y['box_score'] >= self.box_thr] for box_list in annos]
            if self.squeeze:
                annos = [x for x in annos if len(x)]
            item['data'] = annos
            item['total_seconds'] = len(annos)

        samples = []
        self.video_infos_dict = {x['frame_dir']: x for x in self.video_infos}

        for item in self.video_infos:
            annos = item['data']
            num_persons = [len(x) for x in annos]
            samples.extend([
                (item['frame_dir'], idx) for idx in range(len(annos) - self.clip_length + 1)
                if sum(num_persons[idx: idx + self.clip_length]) > 0
            ])

        self.samples = samples

        logger = get_root_logger()
        logger.info(f'{len(self)} samples remain after valid thresholding')

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.samples)

    def load_annotations(self):
        """Load annotation file to get video information."""
        assert self.ann_file.endswith('.pkl')
        data = mload(self.ann_file)

        if self.split:
            split = data['split']
            assert 'annos' in data or 'annotations' in data
            data = data['annos'] if 'annos' in data else data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]

        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])
        return data

    def prepare(self, idx):
        sample = self.samples[idx]
        video = self.video_infos_dict[sample[0]]
        img_shape = video['img_shape']
        clip = cp.deepcopy(video['data'][sample[1]: sample[1] + self.clip_length])
        if self.multi_class:
            for frame in clip:
                for ske in frame:
                    onehot = np.zeros(self.num_classes, dtype=np.float32)
                    label = ske.pop('label', [])
                    onehot[label] = 1.
                    ske['label'] = onehot
        results = {}
        results['frame_dir'] = sample[0]
        results['clip_start'] = sample[1]
        results['clip_length'] = self.clip_length
        results['img_shape'] = img_shape
        results['data'] = clip
        results['modality'] = self.modality
        return results

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = self.prepare(idx)
        results['test_mode'] = False
        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = self.prepare(idx)
        results['test_mode'] = True
        return self.pipeline(results)

    @staticmethod
    def dump_results(results, out):
        score_collect = defaultdict(list)
        for item in results:
            assert isinstance(item, tuple)
            score, names = item[0], item[1]
            assert score.shape[0] == names.shape[0] and len(names.shape) == 1
            for s, n in zip(score, names):
                score_collect[n[:36]].append(s)
        score_collect = {k: np.stack(v).mean(axis=0) for k, v in score_collect.items()}
        return mdump(score_collect, out)

    def evaluate(self, results, logger=None, **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (list): Output results.
            logger (logging.Logger | None): Logger for recording.
                Default: None.

        Returns:
            dict: Evaluation results dict.
        """
        file_name = None
        if not isinstance(results, dict):
            file_name = datetime.now().strftime('%y%m%d_%H%M%S_%f') + '.pkl'
            self.dump_results(results, file_name)
            results = mload(file_name)

        score_collect = results

        names = [k for k in score_collect]
        results = [score_collect[k] for k in score_collect]

        assert len(results[0].shape) in [1, 2]
        if len(results[0].shape) == 1:
            eval_results = ava_map(results, names, self.reweight)
        elif len(results[0].shape) == 2:
            raise NotImplementedError

        if file_name is not None:
            os.remove(file_name)

        return eval_results
