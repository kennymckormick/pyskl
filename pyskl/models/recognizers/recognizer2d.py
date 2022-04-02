# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer2D(BaseRecognizer):
    """2D recognizer model framework."""

    def forward_train(self, imgs, label, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        x = x.reshape((batches, num_segs) + x.shape[1:])

        cls_score = self.cls_head(x)
        gt_label = label.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_label, **kwargs)
        losses.update(loss_cls)

        return losses

    def forward_test(self, imgs, **kwargs):
        """Defines the computation performed at every call when evaluation and testing."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        x = self.extract_feat(imgs)
        assert 'num_segs' in self.test_cfg
        num_segs = self.test_cfg['num_segs']
        assert x.shape[0] % (batches * num_segs) == 0
        num_crops = x.shape[0] // (batches * num_segs)

        if self.test_cfg.get('feat_ext', False):
            # perform spatial pooling
            avg_pool = nn.AdaptiveAvgPool2d(1)
            x = avg_pool(x)
            # squeeze dimensions
            x = x.reshape((batches, num_crops, num_segs, -1))
            # temporal average pooling
            x = x.mean(axis=1).mean(axis=1)
            return x.cpu().numpy()

        x = x.reshape((batches * num_crops, num_segs) + x.shape[1:])
        cls_score = self.cls_head(x)
        cls_score = cls_score.reshape(batches, num_crops, cls_score.shape[-1])
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score)
        return cls_score.cpu().numpy()
