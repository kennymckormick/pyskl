# Copyright (c) OpenMMLab. All rights reserved.
import functools
import numpy as np
import torch
import warnings
from mmcv.runner.hooks import HOOKS, CheckpointHook


@HOOKS.register_module()
class OutputHook:
    """Output feature map of some layers.

    Args:
        module (nn.Module): The whole module to get layers.
        outputs (tuple[str] | list[str]): Layer name to output. Default: None.
        as_tensor (bool): Determine to return a tensor or a numpy array.
            Default: False.
    """

    def __init__(self, module, outputs=None, as_tensor=False):
        self.outputs = outputs
        self.as_tensor = as_tensor
        self.layer_outputs = {}
        self.handles = []
        self.register(module)

    def register(self, module):

        def hook_wrapper(name):

            def hook(model, input, output):
                if not isinstance(output, torch.Tensor):
                    warnings.warn(f'Directly return the output from {name}, '
                                  f'since it is not a tensor')
                    self.layer_outputs[name] = output
                elif self.as_tensor:
                    self.layer_outputs[name] = output
                else:
                    self.layer_outputs[name] = output.detach().cpu().numpy()

            return hook

        if isinstance(self.outputs, (list, tuple)):
            for name in self.outputs:
                try:
                    layer = rgetattr(module, name)
                    h = layer.register_forward_hook(hook_wrapper(name))
                except AttributeError:
                    raise AttributeError(f'Module {name} not found')
                self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()


@HOOKS.register_module()
class MyCkptHook(CheckpointHook):

    def __init__(self, interval=1, **kwargs):
        super().__init__(interval=1, **kwargs)
        if isinstance(interval, int):
            if interval < 1:
                interval = 1
            interval = [(0, np.Inf, interval)]

        assert isinstance(interval, list)
        for i, tup in enumerate(interval):
            assert isinstance(tup, tuple) and len(tup) == 3 and tup[0] < tup[1]
            if i < len(interval) - 1:
                assert tup[1] == interval[i + 1][0]
        assert self.by_epoch
        self.interval = interval

    def every_n_epochs(self, runner, interval):
        cur_epoch = runner.epoch + 1
        for s, e, n in interval:
            if s < cur_epoch <= e:
                return cur_epoch % n == 0 or cur_epoch == e


# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects
def rgetattr(obj, attr, *args):

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))
