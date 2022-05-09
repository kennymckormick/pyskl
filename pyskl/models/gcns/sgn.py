import math

import torch
from mmcv.cnn import ConvModule
from torch import nn

from .utils import unit_sgn


class SGN(nn.Module):

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_joints=25,
                 T=30,
                 bias=True):
        super(SGN, self).__init__()

        self.T = T
        self.num_joints = num_joints
        self.base_channel = base_channels

        self.joint_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.motion_bn = nn.BatchNorm1d(in_channels * num_joints)

        self.t_embed = self.embed_mlp(self.T, base_channels * 4, base_channels, bias=bias)
        self.s_embed = self.embed_mlp(self.num_joints, base_channels, base_channels, bias=bias)
        self.joint_embed = self.embed_mlp(in_channels, base_channels, base_channels, bias=bias)
        self.motion_embed = self.embed_mlp(in_channels, base_channels, base_channels, bias=bias)

        self.compute_A1 = ConvModule(base_channels * 2, base_channels * 4, kernel_size=1, bias=bias)
        self.compute_A2 = ConvModule(base_channels * 2, base_channels * 4, kernel_size=1, bias=bias)

        self.tcn = nn.Sequential(
            nn.AdaptiveMaxPool2d((20, 1)),
            ConvModule(base_channels * 4, base_channels * 4, kernel_size=(3, 1), padding=(1, 0), bias=bias,
                       norm_cfg=dict(type='BN2d')),
            nn.Dropout(0.2),
            ConvModule(base_channels * 4, base_channels * 8, kernel_size=1, bias=bias, norm_cfg=dict(type='BN2d'))
        )

        self.gcn1 = unit_sgn(base_channels * 2, base_channels * 2, bias=bias)
        self.gcn2 = unit_sgn(base_channels * 2, base_channels * 4, bias=bias)
        self.gcn3 = unit_sgn(base_channels * 4, base_channels * 4, bias=bias)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        nn.init.constant_(self.gcn1.conv.weight, 0)
        nn.init.constant_(self.gcn2.conv.weight, 0)
        nn.init.constant_(self.gcn3.conv.weight, 0)

    def embed_mlp(self, in_channels, out_channels, mid_channels=64, bias=False):
        return nn.Sequential(
            ConvModule(in_channels, mid_channels, kernel_size=1, bias=bias),
            ConvModule(mid_channels, out_channels, kernel_size=1, bias=bias),
        )

    def compute_A(self, x):
        # X: N, C, T, V
        A1 = self.compute_A1(x).permute(0, 2, 3, 1).contiguous()
        A2 = self.compute_A2(x).permute(0, 2, 1, 3).contiguous()
        A = A1.matmul(A2)
        return nn.Softmax(dim=-1)(A)

    def forward(self, joint):
        N, M, T, V, C = joint.shape

        joint = joint.reshape(N * M, T, V, C)
        joint = joint.permute(0, 3, 2, 1).contiguous()
        # NM, C, V, T
        motion = torch.diff(joint, dim=3, append=torch.zeros(N * M, C, V, 1).to(joint.device))
        joint = self.joint_bn(joint.view(N * M, C * V, T))
        motion = self.motion_bn(motion.view(N * M, C * V, T))
        joint = joint.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()
        motion = motion.view(N * M, C, V, T).permute(0, 1, 3, 2).contiguous()

        joint_embed = self.joint_embed(joint)
        motion_embed = self.motion_embed(motion)
        # N * M, C, T, V
        t_code = torch.eye(T).to(joint.device)
        t_code = t_code[None, :, None].repeat(N * M, 1, V, 1)
        s_code = torch.eye(V).to(joint.device)
        s_code = s_code[None, ...,  None].repeat(N * M, 1, 1, T)
        t_embed = self.t_embed(t_code).permute(0, 1, 3, 2).contiguous()
        s_embed = self.s_embed(s_code).permute(0, 1, 3, 2).contiguous()

        x = torch.cat([joint_embed + motion_embed, s_embed], 1)
        # N * M, 2base, V, T
        A = self.compute_A(x)
        # N * M, T, V, V
        for gcn in [self.gcn1, self.gcn2, self.gcn3]:
            x = gcn(x, A)

        x = x + t_embed
        x = self.tcn(x)
        # N * M, C, T, V
        return x.reshape((N, M) + x.shape[1:])
