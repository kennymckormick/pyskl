import torch

from ..builder import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class MMRecognizer3D(BaseRecognizer):
    """MultiModality 3D recognizer model framework."""

    def forward_train(self, imgs, heatmap_imgs, label, img_metas=None):
        """Defines the computation performed at every call when training."""
        labels = label
        assert not hasattr(self, 'neck')
        # It's OK, 1 clip training
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        heatmap_imgs = heatmap_imgs.reshape((-1, ) + heatmap_imgs.shape[2:])
        losses = dict()

        # We directly use backbone instead of extract_feats
        # x = self.extract_feat(imgs)
        x_rgb, x_pose = self.backbone(imgs, heatmap_imgs)

        # Which will return 3 cls_scores: ['rgb', 'pose', 'both']
        cls_scores = self.cls_head((x_rgb, x_pose))

        gt_labels = labels.squeeze()
        loss_components = self.cls_head.loss_components
        loss_weights = self.cls_head.loss_weights
        for loss_name, weight in zip(loss_components, loss_weights):
            cls_score = cls_scores[loss_name]
            loss_cls = self.cls_head.loss(cls_score, gt_labels)
            loss_cls = {loss_name + '_' + k: v for k, v in loss_cls.items()}
            loss_cls[f'{loss_name}_loss_cls'] *= weight
            losses.update(loss_cls)
        return losses

    def forward_test(self, imgs, heatmap_imgs, img_metas=None):
        """Defines the computation performed at every call when evaluation and
        testing."""
        assert imgs.shape[0] == 1 and heatmap_imgs.shape[0] == 1
        assert not hasattr(self, 'neck')
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        heatmap_imgs = heatmap_imgs.reshape((-1, ) + heatmap_imgs.shape[2:])

        if self.test_batch is None:
            x_rgb, x_pose = self.backbone(imgs, heatmap_imgs)
            cls_scores = self.cls_head((x_rgb, x_pose))
        else:
            tot = imgs.shape[0]
            ptr = 0
            # will be a list of dict
            cls_scores = []
            while ptr < tot:
                batch_imgs = imgs[ptr:ptr + self.test_batch]
                batch_heatmap_imgs = heatmap_imgs[ptr:ptr + self.test_batch]

                x_rgb, x_pose = self.backbone(batch_imgs, batch_heatmap_imgs)
                cls_scores.append(self.cls_head((x_rgb, x_pose)))
                ptr += self.test_batch

            all_cls_scores = {}
            for k in cls_scores[0]:
                all_cls_scores[k] = torch.cat([item[k] for item in cls_scores])
            cls_scores = all_cls_scores

        for k in cls_scores:
            cls_score = self.average_clip(cls_scores[k], num_segs)
            cls_scores[k] = cls_score.data.cpu().numpy()[0]

        # cuz we use extend for accumulation
        return [cls_scores]

    def forward(self, imgs, heatmap_imgs, label=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        if return_loss:
            if label is None:
                raise ValueError('Label should not be None.')
            return self.forward_train(imgs, heatmap_imgs, label, **kwargs)

        return self.forward_test(imgs, heatmap_imgs, **kwargs)
