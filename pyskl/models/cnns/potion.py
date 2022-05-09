from mmcv.cnn import ConvModule, constant_init, kaiming_init
from torch import nn

from ..builder import BACKBONES


@BACKBONES.register_module()
class PoTion(nn.Module):

    def __init__(self,
                 in_channels,
                 channels=[128, 256, 512],
                 num_layers=[2, 2, 2],
                 lw_dropout=0,
                 conv_cfg=dict(type='Conv'),
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.num_layers = num_layers
        self.lw_dropout = lw_dropout
        assert len(self.channels) == len(self.num_layers)

        layer_names = []
        inplanes = in_channels
        for i, (ch, num_layer) in enumerate(zip(channels, num_layers)):
            layer_name = f'layer{i + 1}'
            layer_names.append(layer_name)
            layer = []
            for j in range(num_layer):
                stride = 2 if j == 0 else 1
                conv = ConvModule(
                    inplanes,
                    ch,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                layer.append(conv)
                if self.lw_dropout > 0:
                    layer.append(nn.Dropout(self.lw_dropout))
                inplanes = ch

            layer = nn.Sequential(*layer)
            setattr(self, layer_name, layer)

        self.layer_names = layer_names

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input samples extracted
            by the backbone.
        """
        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            x = layer(x)

        return x

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
