# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
from collections import defaultdict
from torch.utils.data import DistributedSampler as _DistributedSampler


class DistributedSampler(_DistributedSampler):
    """DistributedSampler inheriting from
    ``torch.utils.data.DistributedSampler``.

    In pytorch of lower versions, there is no ``shuffle`` argument. This child
    class will port one to DistributedSampler.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)


class ClassSpecificDistributedSampler(_DistributedSampler):
    """ClassSpecificDistributedSampler inheriting from 'torch.utils.data.DistributedSampler'.

    Samples are sampled with a class specific probability (class_prob). This sampler is only applicable to single class
    recognition dataset. This sampler is also compatible with RepeatDataset.
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 class_prob=None,
                 shuffle=True,
                 seed=0):

        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        if class_prob is not None:
            if isinstance(class_prob, list):
                class_prob = {i: n for i, n in enumerate(class_prob)}
            assert isinstance(class_prob, dict)
        self.class_prob = class_prob
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        class_prob = self.class_prob
        dataset_name = type(self.dataset).__name__
        dataset = self.dataset if dataset_name != 'RepeatDataset' else self.dataset.dataset
        times = 1
        if dataset_name == 'RepeatDataset':
            times = self.dataset.times
            class_prob = {k: v * times for k, v in class_prob.items()}

        labels = [x['label'] for x in dataset.video_infos]
        samples = defaultdict(list)
        for i, lb in enumerate(labels):
            samples[lb].append(i)

        indices = []
        for class_idx, class_indices in samples.items():
            mul = class_prob.get(class_idx, times)
            for i in range(int(mul // 1)):
                indices.extend(class_indices)
            rem = int((mul % 1) * len(class_indices))
            inds = torch.randperm(len(class_indices), generator=g).tolist()
            indices.extend([class_indices[inds[i]] for i in range(rem)])

        if self.shuffle:
            shuffle = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffle]

        # reset num_samples and total_size here.
        self.num_samples = math.ceil(len(indices) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)
