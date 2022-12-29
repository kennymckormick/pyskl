# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import warnings
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import print_log

from ...utils import cache_checkpoint, get_root_logger
from ..builder import BACKBONES
from .resnet3d import ResNet3d


class DeConvModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=(1, 1, 1),
                 padding=0,
                 bias=False,
                 with_bn=True,
                 with_relu=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.with_bn = with_bn
        self.with_relu = with_relu

        self.conv = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x should be a 5-d tensor
        assert len(x.shape) == 5
        N, C, T, H, W = x.shape
        out_shape = (N, self.out_channels, self.stride[0] * T,
                     self.stride[1] * H, self.stride[2] * W)
        x = self.conv(x, output_size=out_shape)
        if self.with_bn:
            x = self.bn(x)
        if self.with_relu:
            x = self.relu(x)
        return x


class ResNet3dPathway(ResNet3d):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        lateral (bool): Determines whether to enable the lateral connection from another pathway. Default: False.
        speed_ratio (int): Speed ratio indicating the ratio between time dimension of the fast and slow pathway,
            corresponding to the 'alpha' in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway by 'channel_ratio',
            corresponding to 'beta' in the paper. Default: 8.
        fusion_kernel (int): The kernel size of lateral fusion. Default: 7.
        **kwargs (keyword arguments): Keywords arguments for ResNet3d.
    """

    def __init__(self,
                 lateral=False,
                 lateral_inv=False,
                 speed_ratio=8,
                 channel_ratio=8,
                 fusion_kernel=7,
                 lateral_infl=2,
                 lateral_activate=[1, 1, 1, 1],
                 **kwargs):
        self.lateral = lateral
        self.lateral_inv = lateral_inv
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        self.lateral_infl = lateral_infl
        self.lateral_activate = lateral_activate
        self.calculate_lateral_inplanes(kwargs)

        super().__init__(**kwargs)
        self.inplanes = self.base_channels

        if self.lateral and self.lateral_activate[0] == 1:
            if self.lateral_inv:
                self.conv1_lateral = DeConvModule(
                    self.inplanes * self.channel_ratio,
                    self.inplanes * self.channel_ratio // self.lateral_infl,
                    kernel_size=(fusion_kernel, 1, 1),
                    stride=(self.speed_ratio, 1, 1),
                    padding=((fusion_kernel - 1) // 2, 0, 0),
                    with_bn=True,
                    with_relu=True)
            else:
                self.conv1_lateral = ConvModule(
                    self.inplanes // self.channel_ratio,
                    self.inplanes * lateral_infl // self.channel_ratio,
                    kernel_size=(fusion_kernel, 1, 1),
                    stride=(self.speed_ratio, 1, 1),
                    padding=((fusion_kernel - 1) // 2, 0, 0),
                    bias=False,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=None,
                    act_cfg=None)

        self.lateral_connections = []

        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1 and self.lateral_activate[i + 1]:
                # no lateral connection needed in final stage
                lateral_name = f'layer{(i + 1)}_lateral'
                if self.lateral_inv:
                    conv_module = DeConvModule(
                        self.inplanes * self.channel_ratio,
                        self.inplanes * self.channel_ratio // self.lateral_infl,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        with_bn=True,
                        with_relu=True)
                else:
                    conv_module = ConvModule(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * lateral_infl // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=None,
                        act_cfg=None)
                setattr(self, lateral_name, conv_module)
                self.lateral_connections.append(lateral_name)

    def calculate_lateral_inplanes(self, kwargs):
        depth = kwargs.get('depth', 50)
        expansion = 1 if depth < 50 else 4
        base_channels = kwargs.get('base_channels', 64)
        lateral_inplanes = []
        for i in range(kwargs.get('num_stages', 4)):
            if expansion % 2 == 0:
                planes = base_channels * (2 ** i) * ((expansion // 2) ** (i > 0))
            else:
                planes = base_channels * (2 ** i) // (2 ** (i > 0))
            if self.lateral and self.lateral_activate[i]:
                if self.lateral_inv:
                    lateral_inplane = planes * self.channel_ratio // self.lateral_infl
                else:
                    lateral_inplane = planes * self.lateral_infl // self.channel_ratio
            else:
                lateral_inplane = 0
            lateral_inplanes.append(lateral_inplane)
        self.lateral_inplanes = lateral_inplanes

    def inflate_weights(self, logger):
        """Inflate the resnet2d parameters to resnet3d pathway.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart. For pathway the 'lateral_connection' part should
        not be inflated from 2d weights.

        Args:
            logger (logging.Logger): The logger used to print debugging information.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained)
        if 'state_dict' in state_dict_r2d:
            state_dict_r2d = state_dict_r2d['state_dict']

        inflated_param_names = []
        for name, module in self.named_modules():
            if 'lateral' in name:
                continue
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
                    self._inflate_conv_params(module.conv, state_dict_r2d, original_conv_name, inflated_param_names)
                if original_bn_name + '.weight' not in state_dict_r2d:
                    logger.warning(f'Module not exist in the state_dict_r2d: {original_bn_name}')
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d, original_bn_name, inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            logger.info(f'These parameters in the 2d checkpoint are not loaded: {remaining_names}')

    def _inflate_conv_params(self, conv3d, state_dict_2d, module_name_2d, inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        The differences of conv modules betweene 2d and 3d in Pathway
        mainly lie in the inplanes due to lateral connections. To fit the
        shapes of the lateral connection counterpart, it will expand
        parameters by concatting conv2d parameters and extra zero paddings.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the 2d model.
            inflated_param_names (list[str]): List of parameters that have been inflated.
        """
        weight_2d_name = module_name_2d + '.weight'
        conv2d_weight = state_dict_2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]

        if new_shape[1] != old_shape[1]:
            if new_shape[1] < old_shape[1]:
                warnings.warn(f'The parameter of {module_name_2d} is not'
                              'loaded due to incompatible shapes. ')
                return
            # Inplanes may be different due to lateral connections
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels, ) + pad_shape[2:]
            # Expand parameters by concat extra channels
            conv2d_weight = torch.cat(
                (conv2d_weight, torch.zeros(pad_shape).type_as(conv2d_weight).to(conv2d_weight.device)), dim=1)

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, 'bias') is not None:
            bias_2d_name = module_name_2d + '.bias'
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before'self.frozen_stages'. """
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i != len(self.res_layers) and self.lateral:
                # No fusion needed in the final stage
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        super().init_weights()
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)


@BACKBONES.register_module()
class ResNet3dSlowFast(nn.Module):
    """Slowfast backbone.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        resample_rate (int): A large temporal stride 'resample_rate' on input frames. The actual resample rate is
            calculated by multipling the 'interval' in 'SampleFrames' in the pipeline with 'resample_rate', equivalent
            to the :math:`\\tau` in the paper, i.e. it processes only one out of 'resample_rate * interval' frames.
            Default: 8.
        speed_ratio (int): Speed ratio indicating the ratio between time dimension of the fast and slow pathway,
            corresponding to the :math:`\\alpha` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway by 'channel_ratio', corresponding to
            :math:`\\beta` in the paper. Default: 8.
        slow_pathway (dict): Configuration of slow branch.
            Default: dict(lateral=True, depth=50, conv1_kernel=(1, 7, 7), inflate=(0, 0, 1, 1))
        fast_pathway (dict): Configuration of fast branch.
            Default: dict(lateral=False, depth=50, base_channels=8, conv1_kernel=(5, 7, 7))
    """

    def __init__(self,
                 pretrained=None,
                 resample_rate=8,
                 speed_ratio=8,
                 channel_ratio=8,
                 slow_pathway=dict(
                     depth=50,
                     lateral=True,
                     conv1_kernel=(1, 7, 7),
                     inflate=(0, 0, 1, 1)),
                 fast_pathway=dict(
                     depth=50,
                     lateral=False,
                     base_channels=8,
                     conv1_kernel=(5, 7, 7))):
        super().__init__()
        self.pretrained = pretrained
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if slow_pathway['lateral']:
            slow_pathway['speed_ratio'] = speed_ratio
            slow_pathway['channel_ratio'] = channel_ratio

        self.slow_path = ResNet3dPathway(**slow_pathway)
        self.fast_path = ResNet3dPathway(**fast_pathway)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            msg = f'load model from: {self.pretrained}'
            print_log(msg, logger=logger)
            # Directly load 3D model.
            self.pretrained = cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=True, logger=logger)
        elif self.pretrained is None:
            # Init two branch separately.
            self.fast_path.init_weights()
            self.slow_path.init_weights()
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted by the backbone.
        """
        x_slow = nn.functional.interpolate(
            x, mode='nearest', scale_factor=(1.0 / self.resample_rate, 1.0, 1.0))
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)

        x_fast = nn.functional.interpolate(
            x, mode='nearest', scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0, 1.0))
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        if self.slow_path.lateral:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)
            if (i != len(self.slow_path.res_layers) - 1 and self.slow_path.lateral):
                # No fusion needed in the final stage
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        out = (x_slow, x_fast)

        return out
