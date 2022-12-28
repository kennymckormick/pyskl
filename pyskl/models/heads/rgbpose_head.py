import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


@HEADS.register_module()
class RGBPoseHead(BaseHead):
    """The classification head for Slowfast.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (tuple[int]): Number of channels in input feature.
        loss_cls (dict): Config for building loss. Default: dict(type='CrossEntropyLoss').
        dropout (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initializ the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 loss_components=['rgb', 'pose'],
                 loss_weights=1.,
                 dropout=0.5,
                 init_std=0.01,
                 **kwargs):

        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        if isinstance(dropout, float):
            dropout = {'rgb': dropout, 'pose': dropout}
        assert isinstance(dropout, dict)

        self.dropout = dropout
        self.init_std = init_std
        self.in_channels = in_channels

        self.loss_components = loss_components
        if isinstance(loss_weights, float):
            loss_weights = [loss_weights] * len(loss_components)

        assert len(loss_weights) == len(loss_components)
        self.loss_weights = loss_weights

        self.dropout_rgb = nn.Dropout(p=self.dropout['rgb'])
        self.dropout_pose = nn.Dropout(p=self.dropout['pose'])

        self.fc_rgb = nn.Linear(in_channels[0], num_classes)
        self.fc_pose = nn.Linear(in_channels[1], num_classes)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_rgb, std=self.init_std)
        normal_init(self.fc_pose, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        x_rgb, x_pose = self.avg_pool(x[0]), self.avg_pool(x[1])
        x_rgb = x_rgb.view(x_rgb.size(0), -1)
        x_pose = x_pose.view(x_pose.size(0), -1)

        x_rgb = self.dropout_rgb(x_rgb)
        x_pose = self.dropout_pose(x_pose)

        cls_scores = {}
        cls_scores['rgb'] = self.fc_rgb(x_rgb)
        cls_scores['pose'] = self.fc_pose(x_pose)

        return cls_scores
