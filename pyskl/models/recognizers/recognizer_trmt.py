import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from collections import OrderedDict

from .. import builder
from ..builder import RECOGNIZERS
from .base import BaseRecognizer

EPS = 1e-4


@RECOGNIZERS.register_module()
class RecognizerTRMT(BaseRecognizer):
    """SkeleTR wrapper, which supports multi-task training for skeleton-based action understanding. """

    def __init__(self, backbone, cls_head=None, train_cfg=dict(), test_cfg=dict(), flexible_nske=None):
        super(BaseRecognizer, self).__init__()
        # record the source of the backbone
        self.backbone = builder.build_backbone(backbone)

        # If head == MetaHead, then meta_head takes over the post-processing, x[:, 0] for Kinetics, x[:, 1:] for AVA
        # otherwise, the Recognizer takes over the post-processing
        self.cls_head = builder.build_head(cls_head) if cls_head else None

        assert isinstance(train_cfg, dict) and isinstance(test_cfg, dict)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights()
        # It only applies to training
        if flexible_nske is not None:
            if flexible_nske == 'random_select':
                pass
            else:
                assert len(flexible_nske) == 2 and 0 < flexible_nske[0] <= flexible_nske[1]
        self.flexible_nske = flexible_nske

    def keep_n_skeletons(self, kpts, stis, labs, n_skeletons):
        # Only used in training
        assert len(kpts.shape) == 5 and kpts.shape[1] >= n_skeletons  # N, M, T, V, C
        assert len(stis.shape) == 3 and stis.shape[1] >= n_skeletons  # N, M, 6
        assert kpts.shape[1] == stis.shape[1] and kpts.shape[0] == stis.shape[0] == len(labs)
        num_skeletons = kpts.shape[1]
        keypoints, stinfo = [], []
        for kpt, sti, lab in zip(kpts, stis, labs):
            # M, T, V, C; M, 6; 1
            non_zero = kpt[..., 2].sum(dim=[1, 2]) > EPS
            nz_indices = [i for i, k in enumerate(non_zero) if k]
            z_indices = [i for i, k in enumerate(non_zero) if not k]
            if len(nz_indices) > n_skeletons:
                indices = np.sort(np.random.choice(nz_indices, n_skeletons, replace=False))
            else:
                indices = nz_indices + z_indices[:n_skeletons - len(nz_indices)]
                indices.sort()
            keypoints.append(kpt[indices])
            stinfo.append(sti[indices])

            for k, v in lab.items():
                if len(v.shape) == 3:
                    assert v.shape[1] == num_skeletons and v.shape[0] == 1
                    lab[k] = v[:, indices]

        keypoints, stinfo = torch.stack(keypoints), torch.stack(stinfo)
        return keypoints, stinfo, labs

    def get_n_skeletons(self, keypoint):
        assert len(keypoint.shape) == 5
        batch = keypoint.shape[0]
        nums = []
        for i in range(batch):
            non_zero = keypoint[i, ..., 2].sum(dim=[1, 2]) > EPS
            non_zero = non_zero.data.cpu().numpy()
            nums.append(non_zero.sum())
        return nums

    def forward_train(self, keypoint, stinfo, img_metas):
        """Defines the computation performed at every call when training."""
        assert self.with_cls_head
        assert keypoint.shape[1] == 1

        keypoint = keypoint[:, 0]

        if stinfo is not None:
            stinfo = stinfo[:, 0]

        assert len(img_metas) == keypoint.shape[0]

        # meta['label'] is a dict: {label_name: label_tensor}
        labels = [meta['label'] for meta in img_metas]

        if self.flexible_nske:
            if self.flexible_nske == 'random_select':
                nums = self.get_n_skeletons(keypoint)
                nske = np.random.choice(nums, 1)[0]
            else:
                min_ske, max_ske = self.flexible_nske
                nske = np.random.choice(range(min_ske, max_ske + 1), 1)[0]
            keypoint, stinfo, labels = self.keep_n_skeletons(keypoint, stinfo, labels, nske)

        # If cls_token is not None, will use cls_token for video-level classification
        cls_token, x = self.extract_feat(keypoint, stinfo)  # N, C; N, M, C
        losses = dict()

        meta_head = getattr(self.cls_head, 'meta_head', False)
        if meta_head:
            # meta['tag'] is a list of tags, it includes all tasks associated with the training sample (has GT labels)
            tags = [meta['tag'] for meta in img_metas]
            # cls_score is a list of dict: {task_name: cls_score}
            cls_score = self.cls_head(cls_token, x, tags)
            loss = self.cls_head.loss(cls_score, labels, tags)
        else:
            # Single Task Sanity Check
            if 'tag' in img_metas[0]:
                assert len(img_metas[0]['tag']) == 1
                tag = img_metas[0]['tag'][0]
                for meta in img_metas:
                    assert len(meta['tag']) == 1 and meta['tag'][0] == tag
            assert len(labels[0]) == 1
            label_name = list(labels[0].keys())[0]
            for label in labels:
                assert len(label) == 1 and label_name in label
            # Get Label Tensors
            labels = [x[label_name] for x in labels]
            label = torch.stack(labels)

            if len(label.shape) == 4:
                label = label[:, 0]
                cls_score = self.cls_head(x)
            elif len(label.shape) == 2:
                label = label.squeeze(-1)
                cls_score = self.cls_head(cls_token if cls_token is not None else x)
                if len(cls_score.shape) == 3:
                    cls_score = cls_score.mean(dim=1)
            loss = self.cls_head.loss(cls_score, label)
        losses.update(loss)
        return losses

    def forward_test(self, keypoint, stinfo, img_metas):
        """Defines the computation performed at every call when evaluation and testing."""
        assert self.with_cls_head
        # The shape here is Batch, Aug, Ske, T, V, C
        bs, nc = keypoint.shape[:2]
        assert bs == 1
        keypoint = keypoint[0]

        if stinfo is not None:
            stinfo = stinfo[0]

        assert len(img_metas) == 1

        assert 'name' in img_metas[0] or 'frame_dir' in img_metas[0]
        name = img_metas[0].get('name', img_metas[0]['frame_dir'])
        cls_token, x = self.extract_feat(keypoint, stinfo)  # Aug, C; Aug, M, C

        tags = img_metas[0]['tag']

        meta_head = getattr(self.cls_head, 'meta_head', False)
        if meta_head:
            result = dict()
            cls_score = self.cls_head(cls_token, x, tags)
            if isinstance(name, dict):
                for tag, n in name.items():
                    if tag in cls_score:
                        score = cls_score[tag]
                        if isinstance(n, str):
                            assert len(score.shape) == 2
                            score = nn.Softmax(dim=-1)(score).mean(dim=0)[None].data.cpu().numpy().astype(np.float16)
                            result[tag] = (n, score)
                        elif isinstance(n, np.ndarray):
                            assert len(score.shape) == 3
                            score = nn.Sigmoid()(score).data.cpu().numpy().astype(np.float16)
                            result[tag] = (n, score)
                return [result]
            elif isinstance(name, str):
                assert len(cls_score) == 1 and len(tags) == 1
                result = cls_score[tags[0]]
                assert len(result.shape) == 2
                result = nn.Softmax(dim=-1)(result).mean(dim=0)[None].data.cpu().numpy().astype(np.float16)
                return [(name, result)]
            elif isinstance(name, np.ndarray):
                assert len(cls_score) == 1 and len(tags) == 1
                result = cls_score[tags[0]]
                assert len(result.shape) == 3
                result = nn.Sigmoid()(result).data.cpu().numpy().astype(np.float16)
                return [(name, result)]
            else:
                raise NotImplementedError
        else:
            assert not isinstance(name, dict)
            if isinstance(name, np.ndarray):
                cls_score = self.cls_head(x)
                cls_score = nn.Sigmoid()(cls_score).data.cpu().numpy().astype(np.float16)
                return [(name, cls_score)]
            elif isinstance(name, str):
                cls_score = self.cls_head(cls_token if cls_token is not None else x)
                if len(cls_score.shape) == 3:
                    cls_score = cls_score.mean(dim=1)
                cls_score = nn.Softmax(dim=-1)(cls_score).mean(dim=0)[None].data.cpu().numpy().astype(np.float16)
                return [(name, cls_score)]
            else:
                raise NotImplementedError

    def forward(self, keypoint, stinfo=None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        assert 'img_metas' in kwargs
        if return_loss:
            return self.forward_train(keypoint, stinfo, kwargs['img_metas'])

        return self.forward_test(keypoint, stinfo, kwargs['img_metas'])

    def extract_feat(self, keypoint, stinfo=None):
        return self.backbone(keypoint, stinfo)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        meta_head = getattr(self.cls_head, 'meta_head', False)
        if meta_head:
            log_vars = OrderedDict()
            for loss_name, loss_value in losses.items():
                if 'acc' in loss_name:
                    continue
                if isinstance(loss_value, torch.Tensor):
                    log_vars[loss_name] = loss_value.mean()
                elif isinstance(loss_value, list):
                    log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
                else:
                    raise TypeError(f'{loss_name} is not a tensor or list of tensors')

            tags = self.cls_head.head_names
            total, loss = 0, 0
            for tag in tags:
                keys = [k for k in log_vars if tag in k and 'num' not in k]
                if len(keys) == 0:
                    continue
                num = log_vars[f'{tag}_num']
                total = num + total
                for k in keys:
                    log_vars[k] = log_vars[k] * num
                loss = log_vars[f'{tag}_loss_cls'] + loss
            loss = loss / total
            log_vars['loss'] = loss
            for loss_name, loss_value in log_vars.items():
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value)
                log_vars[loss_name] = loss_value.item()
            log_vars['loss'] = log_vars['loss'] / dist.get_world_size()
            for tag in tags:
                keys = [k for k in log_vars if tag in k and 'num' not in k]
                if len(keys) == 0:
                    continue
                num = log_vars.pop(f'{tag}_num')
                for k in keys:
                    log_vars[k] = log_vars[k] / (num + EPS)
            return loss, log_vars
        else:
            return super()._parse_losses(losses)
