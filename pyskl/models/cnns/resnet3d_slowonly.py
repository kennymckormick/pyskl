# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import BACKBONES
from .resnet3d import ResNet3d


@BACKBONES.register_module()
class ResNet3dSlowOnly(ResNet3d):
    """SlowOnly backbone based on ResNet3d.

    Args:
        conv1_kernel (tuple[int]): Kernel size of the first conv layer. Default: (1, 7, 7).
        inflate (tuple[int]): Inflate Dims of each block. Default: (0, 0, 1, 1).
        **kwargs (keyword arguments): Other keywords arguments for 'ResNet3d'.
    """

    def __init__(self, conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1), **kwargs):
        super().__init__(conv1_kernel=conv1_kernel, inflate=inflate, **kwargs)
