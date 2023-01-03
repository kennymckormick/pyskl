import torch
import torch.nn as nn
from abc import ABCMeta

from ..builder import HEADS, build_head


@HEADS.register_module()
class MetaHead(nn.Module, metaclass=ABCMeta):
    """ A simple classification head.

    Args:
        **kwargs: head_name (key), head_cfg (value)
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.use_cls_token = dict()
        self.global_flag = dict()

        for k in kwargs:
            assert isinstance(kwargs[k], dict)
            assert 'type' in kwargs[k]
            self.use_cls_token[k] = kwargs[k].pop('use_cls_token', False)
            self.global_flag[k] = kwargs[k].pop('global_flag', True)

        heads = dict()
        for k in kwargs:
            heads[k] = build_head(kwargs[k])
        self.head_names = [k for k in heads]
        self.heads = nn.ModuleDict(heads)
        self.meta_head = True

    def init_weights(self):
        for k in self.head_names:
            self.heads[k].init_weights()

    # Tags is a one-level list or two-level list
    def forward(self, cls_token, x, tags):
        assert isinstance(tags, list)
        tag_level = 1 if isinstance(tags[0], str) else 2

        # That's testing mode
        if tag_level == 1:
            cls_score = dict()
            for tag in tags:
                assert tag in self.head_names
                inp = cls_token if (self.use_cls_token[tag] and cls_token is not None) else x
                score = self.heads[tag](inp)
                if self.global_flag[tag] and len(score.shape) == 3:
                    score = score.mean(dim=1)
                cls_score[tag] = score
            return cls_score

        for tag_list in tags:
            for tag in tag_list:
                assert tag in self.head_names

        outs = [dict() for _ in tags]

        for tag in self.head_names:
            indices = [i for i, tag_list in enumerate(tags) if tag in tag_list]
            if len(indices):
                inp = cls_token if (self.use_cls_token[tag] and cls_token is not None) else x
                cls_score = self.heads[tag](inp[indices])
                if self.global_flag[tag] and len(cls_score.shape) == 3:
                    cls_score = cls_score.mean(dim=1)
                for i, s in zip(indices, cls_score):
                    outs[i][tag] = s
        return outs

    # That is called only in training mode
    def loss(self, cls_score, label, tags):
        assert isinstance(tags, list) and isinstance(tags[0], list)
        for tag_list in tags:
            for tag in tag_list:
                assert tag in self.head_names

        losses = dict()
        for tag in self.head_names:
            indices = [i for i, tag_list in enumerate(tags) if tag in tag_list]
            if len(indices) == 0:
                zero = torch.tensor(0).type(torch.float32).to(cls_score[0].device)
                losses[f'{tag}_loss_cls'] = zero
                losses[f'{tag}_num'] = zero
                if not self.heads[tag].multi_class:
                    losses[f'{tag}_top1_acc'] = zero
                    losses[f'{tag}_top5_acc'] = zero
                continue

            scores = [cls_score[i][tag] for i in indices]
            scores = torch.stack(scores)
            labels = [label[i][tag] for i in indices]
            labels = torch.stack(labels)[:, 0]

            sub_losses = self.heads[tag].loss(scores, labels)
            sub_losses = {f'{tag}_{k}': sub_losses[k] for k in sub_losses}
            losses.update(sub_losses)
            losses[f'{tag}_num'] = torch.tensor(len(indices)).type(torch.float32).to(cls_score[0].device)

        return losses
