# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import warnings
from mmcv.cnn import ConvModule, build_activation_layer, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.nn.modules.utils import _ntuple, _triple

from ...utils import cache_checkpoint, get_root_logger
from ..builder import BACKBONES


class BasicBlock3d(nn.Module):
    """BasicBlock 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        stride (tuple): Stride is a two element tuple (temporal, spatial). Default: (1, 1).
        downsample (nn.Module | None): Downsample layer. Default: None.
        inflate (bool): Whether to inflate kernel. Default: True.
        conv_cfg (dict): Config dict for convolution layer. Default: 'dict(type='Conv3d')'.
        norm_cfg (dict): Config for norm layers. required keys are 'type'. Default: 'dict(type='BN3d')'.
        act_cfg (dict): Config dict for activation layer. Default: 'dict(type='ReLU')'.
    """
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=(1, 1),
                 downsample=None,
                 inflate=True,
                 inflate_style='3x3x3',
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        assert inflate_style == '3x3x3'

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv1 = ConvModule(
            inplanes,
            planes,
            3 if self.inflate else (1, 3, 3),
            stride=(self.stride[0], self.stride[1], self.stride[1]),
            padding=1 if self.inflate else (0, 1, 1),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            3 if self.inflate else (1, 3, 3),
            stride=1,
            padding=1 if self.inflate else (0, 1, 1),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out


class Bottleneck3d(nn.Module):
    """Bottleneck 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        stride (tuple): Stride is a two element tuple (temporal, spatial). Default: (1, 1).
        downsample (nn.Module | None): Downsample layer. Default: None.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): '3x1x1' or '3x3x3'. which determines the kernel sizes and padding strides
            for conv1 and conv2 in each block. Default: '3x1x1'.
        conv_cfg (dict): Config dict for convolution layer. Default: 'dict(type='Conv3d')'.
        norm_cfg (dict): Config for norm layers. required keys are 'type'. Default: 'dict(type='BN3d')'.
        act_cfg (dict): Config dict for activation layer. Default: 'dict(type='ReLU')'.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=(1, 1),
                 downsample=None,
                 inflate=True,
                 inflate_style='3x1x1',
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d'),
                 act_cfg=dict(type='ReLU')):
        super().__init__()
        assert inflate_style in ['3x1x1', '3x3x3']

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg

        mode = 'no_inflate' if not self.inflate else self.inflate_style
        conv1_kernel_size = {'no_inflate': 1, '3x1x1': (3, 1, 1), '3x3x3': 1}
        conv1_padding = {'no_inflate': 0, '3x1x1': (1, 0, 0), '3x3x3': 0}
        conv2_kernel_size = {'no_inflate': (1, 3, 3), '3x1x1': (1, 3, 3), '3x3x3': 3}
        conv2_padding = {'no_inflate': (0, 1, 1), '3x1x1': (0, 1, 1), '3x3x3': 1}

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size[mode],
            stride=1,
            padding=conv1_padding[mode],
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size[mode],
            stride=(self.stride[0], self.stride[1], self.stride[1]),
            padding=conv2_padding[mode],
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            # No activation in the third ConvModule for bottleneck
            act_cfg=None)

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        out = _inner_forward(x)
        out = self.relu(out)

        return out


@BACKBONES.register_module()
class ResNet3d(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}. Default: 50.
        pretrained (str | None): Name of pretrained model.
        stage_blocks (tuple | None): Set number of stages for each res layer. Default: None.
        pretrained2d (bool): Whether to load pretrained 2D model. Default: True.
        in_channels (int): Channel num of input features. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
        out_indices (tuple[int]): Indices of output feature. Default: (3, ).
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (tuple[int]): Spatial strides of residual blocks of each stage. Default: (1, 2, 2, 2).
        temporal_strides (tuple[int]): Temporal strides of residual blocks of each stage. Default: (1, 1, 1, 1).
        conv1_kernel (tuple[int]): Kernel size of the first conv layer. Default: (3, 7, 7).
        conv1_stride (tuple[int]): Stride of the first conv layer (temporal, spatial). Default: (1, 2).
        pool1_stride (tuple[int]): Stride of the first pooling layer (temporal, spatial). Default: (1, 2).
        advanced (bool): Flag indicating if an advanced design for downsample is adopted. Default: False.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means not freezing any parameters. Default: -1.
        inflate (tuple[int]): Inflate Dims of each block. Default: (1, 1, 1, 1).
        inflate_style (str): '3x1x1' or '3x3x3'. which determines the kernel sizes and padding strides
            for conv1 and conv2 in each block. Default: '3x1x1'.
        conv_cfg (dict): Config for conv layers. required keys are 'type'. Default: 'dict(type='Conv3d')'.
        norm_cfg (dict): Config for norm layers. required keys are 'type' and 'requires_grad'.
            Default: 'dict(type='BN3d', requires_grad=True)'.
        act_cfg (dict): Config dict for activation layer. Default: 'dict(type='ReLU', inplace=True)'.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze running stats (mean and var).
            Default: False.
        zero_init_residual (bool): Whether to use zero initialization for residual block. Default: True.
    """

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d, (3, 4, 6, 3)),
        101: (Bottleneck3d, (3, 4, 23, 3)),
        152: (Bottleneck3d, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth=50,
                 pretrained=None,
                 stage_blocks=None,
                 pretrained2d=True,
                 in_channels=3,
                 num_stages=4,
                 base_channels=64,
                 out_indices=(3, ),
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 conv1_kernel=(3, 7, 7),
                 conv1_stride=(1, 2),
                 pool1_stride=(1, 2),
                 advanced=False,
                 frozen_stages=-1,
                 inflate=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 conv_cfg=dict(type='Conv3d'),
                 norm_cfg=dict(type='BN3d', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_eval=False,
                 zero_init_residual=True):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        assert len(spatial_strides) == len(temporal_strides) == num_stages
        if self.stage_blocks is not None:
            assert len(self.stage_blocks) == num_stages

        self.conv1_kernel = conv1_kernel
        self.conv1_stride = conv1_stride
        self.pool1_stride = pool1_stride
        self.advanced = advanced
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]

        if self.stage_blocks is None:
            self.stage_blocks = stage_blocks[:num_stages]

        self.inplanes = self.base_channels

        self._make_stem_layer()
        self.res_layers = []
        # This field can be utilized by ResNet3dPathway, and has not side effect.
        lateral_inplanes = getattr(self, 'lateral_inplanes', [0, 0, 0, 0])

        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            planes = self.base_channels * 2**i
            res_layer = self.make_res_layer(
                self.block,
                self.inplanes + lateral_inplanes[i],
                planes,
                num_blocks,
                stride=(temporal_stride, spatial_stride),
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                advanced=self.advanced,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2 ** (len(self.stage_blocks) - 1)

    @staticmethod
    def make_res_layer(block,
                       inplanes,
                       planes,
                       blocks,
                       stride=(1, 1),
                       inflate=1,
                       inflate_style='3x1x1',
                       advanced=False,
                       norm_cfg=None,
                       act_cfg=None,
                       conv_cfg=None):
        """Build residual layer for ResNet3D.

        Args:
            block (nn.Module): Residual module to be built.
            inplanes (int): Number of channels for the input feature in each block.
            planes (int): Number of channels for the output feature in each block.
            blocks (int): Number of residual blocks.
            stride (tuple[int]): Stride (temporal, spatial) in residual and conv layers. Default: (1, 1).
            inflate (int | tuple[int]): Determine whether to inflate for each block. Default: 1.
            inflate_style (str): '3x1x1' or '3x3x3'. which determines the kernel sizes and padding strides
                for conv1 and conv2 in each block. Default: '3x1x1'.
            conv_cfg (dict | None): Config for norm layers. Default: None.
            norm_cfg (dict | None): Config for norm layers. Default: None.
            act_cfg (dict | None): Config for activate layers. Default: None.

        Returns:
            nn.Module: A residual layer for the given config.
        """
        inflate = inflate if not isinstance(inflate, int) else (inflate, ) * blocks
        assert len(inflate) == blocks
        downsample = None
        if stride[1] != 1 or inplanes != planes * block.expansion:
            if advanced:
                conv = ConvModule(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None)
                pool = nn.AvgPool3d(
                    kernel_size=(stride[0], stride[1], stride[1]),
                    stride=(stride[0], stride[1], stride[1]),
                    ceil_mode=True)
                downsample = nn.Sequential(conv, pool)
            else:
                downsample = ConvModule(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(stride[0], stride[1], stride[1]),
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=None)

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                norm_cfg=norm_cfg,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                    stride=(1, 1),
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg))

        return nn.Sequential(*layers)

    @staticmethod
    def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d, inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the 2d model.
            inflated_param_names (list[str]): List of parameters that have been inflated.
        """
        weight_2d_name = module_name_2d + '.weight'

        conv2d_weight = state_dict_2d[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    @staticmethod
    def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d,
                           inflated_param_names):
        """Inflate a norm module from 2d to 3d.

        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding bn module in the 2d model.
            inflated_param_names (list[str]): List of parameters that have been inflated.
        """
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f'{module_name_2d}.{param_name}'
            param_2d = state_dict_2d[param_2d_name]
            if param.data.shape != param_2d.shape:
                warnings.warn(f'The parameter of {module_name_2d} is not loaded due to incompatible shapes. ')
                return

            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)

        for param_name, param in bn3d.named_buffers():
            param_2d_name = f'{module_name_2d}.{param_name}'
            # some buffers like num_batches_tracked may not exist in old checkpoints
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    @staticmethod
    def _inflate_weights(self, logger):
        """Inflate the resnet2d parameters to resnet3d.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the name mapping is needed
                if 'downsample' in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + '.0'
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + '.1'
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace('conv', 'bn')
                if original_conv_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d: {original_conv_name}')
                else:
                    shape_2d = state_dict_r2d[original_conv_name + '.weight'].shape
                    shape_3d = module.conv.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        logger.warning(f'Weight shape mismatch for: {original_conv_name}: '
                                       f'3d weight shape: {shape_3d}; 2d weight shape: {shape_2d}.')
                    else:
                        self._inflate_conv_params(
                            module.conv, state_dict_r2d, original_conv_name, inflated_param_names
                        )

                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d: {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d, original_bn_name, inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded: {remaining_names}')

    def inflate_weights(self, logger):
        self._inflate_weights(self, logger)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride[0], self.conv1_stride[1], self.conv1_stride[1]),
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride[0], self.pool1_stride[1], self.pool1_stride[1]),
            padding=(0, 1, 1))

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        'self.frozen_stages'."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @staticmethod
    def _init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will override the original 'pretrained' if set.
                The arg is added to be compatible with mmdet. Default: None.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3d):
                    constant_init(m.conv3.bn, 0)
                elif isinstance(m, BasicBlock3d):
                    constant_init(m.conv2.bn, 0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            if self.pretrained2d:
                self.inflate_weights(logger)
            else:
                self.pretrained = cache_checkpoint(self.pretrained)
                load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def init_weights(self, pretrained=None):
        self._init_weights(self, pretrained)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
