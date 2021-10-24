"""
    RegNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.
"""

__all__ = ['RegNet', 'regnetx002', 'regnetx004', 'regnetx006', 'regnetx008', 'regnetx016', 'regnetx032', 'regnetx040',
           'regnetx064', 'regnetx080', 'regnetx120', 'regnetx160', 'regnetx320', 'regnety002', 'regnety004',
           'regnety006', 'regnety008', 'regnety016', 'regnety032', 'regnety040', 'regnety064', 'regnety080',
           'regnety120', 'regnety160', 'regnety320']

import os
import numpy as np
import torch.nn as nn
from .common import conv1x1_block, conv3x3_block, SEBlock


class RegNetBottleneck(nn.Module):
    """
    RegNet bottleneck block for residual path in RegNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    groups : int
        Number of groups.
    use_se : bool
        Whether to use SE-module.
    bottleneck_factor : int, default 1
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups,
                 use_se,
                 bottleneck_factor=1):
        super(RegNetBottleneck, self).__init__()
        self.use_se = use_se
        mid_channels = out_channels // bottleneck_factor
        mid_groups = mid_channels // groups

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            groups=mid_groups)
        if self.use_se:
            self.se = SEBlock(
                channels=mid_channels,
                mid_channels=(in_channels // 4))
        self.conv3 = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv3(x)
        return x


class RegNetUnit(nn.Module):
    """
    RegNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    groups : int
        Number of groups.
    use_se : bool
        Whether to use SE-module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups,
                 use_se):
        super(RegNetUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = RegNetBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            groups=groups,
            use_se=use_se)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        return x


class RegNet(nn.Module):
    """
    RegNet model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    groups : list of int
        Number of groups for each stage.
    use_se : bool
        Whether to use SE-module.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 groups,
                 use_se,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(RegNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            padding=1))
        in_channels = init_block_channels
        for i, (channels_per_stage, groups_per_stage) in enumerate(zip(channels, groups)):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) else 1
                stage.add_module("unit{}".format(j + 1), RegNetUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    groups=groups_per_stage,
                    use_se=use_se))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(output_size=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_regnet(channels_init,
               channels_slope,
               channels_mult,
               depth,
               groups,
               use_se=False,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create RegNet model with specific parameters.

    Parameters:
    ----------
    channels_init : float
        Initial value for channels/widths.
    channels_slope : float
        Slope value for channels/widths.
    width_mult : float
        Width multiplier value.
    groups : int
        Number of groups.
    depth : int
        Depth value.
    use_se : bool, default False
        Whether to use SE-module.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    divisor = 8
    assert (channels_slope >= 0) and (channels_init > 0) and (channels_mult > 1) and (channels_init % divisor == 0)

    # Generate continuous per-block channels/widths:
    channels_cont = np.arange(depth) * channels_slope + channels_init

    # Generate quantized per-block channels/widths:
    channels_exps = np.round(np.log(channels_cont / channels_init) / np.log(channels_mult))
    channels = channels_init * np.power(channels_mult, channels_exps)
    channels = (np.round(channels / divisor) * divisor).astype(np.int)

    # Generate per stage channels/widths and layers/depths:
    channels_per_stage, layers = np.unique(channels, return_counts=True)

    # Adjusts the compatibility of channels/widths and groups:
    groups_per_stage = [min(groups, c) for c in channels_per_stage]
    channels_per_stage = [int(round(c / g) * g) for c, g in zip(channels_per_stage, groups_per_stage)]

    channels = [[ci] * li for (ci, li) in zip(channels_per_stage, layers)]

    init_block_channels = 32

    net = RegNet(
        channels=channels,
        init_block_channels=init_block_channels,
        groups=groups_per_stage,
        use_se=use_se,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def regnetx002(**kwargs):
    """
    RegNetX-200MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=24, channels_slope=36.44, channels_mult=2.49, depth=13, groups=8,
                      model_name="regnetx002", **kwargs)


def regnetx004(**kwargs):
    """
    RegNetX-400MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=24, channels_slope=24.48, channels_mult=2.54, depth=22, groups=16,
                      model_name="regnetx004", **kwargs)


def regnetx006(**kwargs):
    """
    RegNetX-600MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=48, channels_slope=36.97, channels_mult=2.24, depth=16, groups=24,
                      model_name="regnetx006", **kwargs)


def regnetx008(**kwargs):
    """
    RegNetX-800MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=56, channels_slope=35.73, channels_mult=2.28, depth=16, groups=16,
                      model_name="regnetx008", **kwargs)


def regnetx016(**kwargs):
    """
    RegNetX-1.6GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=80, channels_slope=34.01, channels_mult=2.25, depth=18, groups=24,
                      model_name="regnetx016", **kwargs)


def regnetx032(**kwargs):
    """
    RegNetX-3.2GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=88, channels_slope=26.31, channels_mult=2.25, depth=25, groups=48,
                      model_name="regnetx032", **kwargs)


def regnetx040(**kwargs):
    """
    RegNetX-4.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=96, channels_slope=38.65, channels_mult=2.43, depth=23, groups=40,
                      model_name="regnetx040", **kwargs)


def regnetx064(**kwargs):
    """
    RegNetX-6.4GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=184, channels_slope=60.83, channels_mult=2.07, depth=17, groups=56,
                      model_name="regnetx064", **kwargs)


def regnetx080(**kwargs):
    """
    RegNetX-8.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=80, channels_slope=49.56, channels_mult=2.88, depth=23, groups=120,
                      model_name="regnetx080", **kwargs)


def regnetx120(**kwargs):
    """
    RegNetX-12GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=168, channels_slope=73.36, channels_mult=2.37, depth=19, groups=112,
                      model_name="regnetx120", **kwargs)


def regnetx160(**kwargs):
    """
    RegNetX-16GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=216, channels_slope=55.59, channels_mult=2.1, depth=22, groups=128,
                      model_name="regnetx160", **kwargs)


def regnetx320(**kwargs):
    """
    RegNetX-32GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=320, channels_slope=69.86, channels_mult=2.0, depth=23, groups=168,
                      model_name="regnetx320", **kwargs)


def regnety002(**kwargs):
    """
    RegNetY-200MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=24, channels_slope=36.44, channels_mult=2.49, depth=13, groups=8, use_se=True,
                      model_name="regnety002", **kwargs)


def regnety004(**kwargs):
    """
    RegNetY-400MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=48, channels_slope=27.89, channels_mult=2.09, depth=16, groups=8, use_se=True,
                      model_name="regnety004", **kwargs)


def regnety006(**kwargs):
    """
    RegNetY-600MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=48, channels_slope=32.54, channels_mult=2.32, depth=15, groups=16, use_se=True,
                      model_name="regnety006", **kwargs)


def regnety008(**kwargs):
    """
    RegNetY-800MF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=56, channels_slope=38.84, channels_mult=2.4, depth=14, groups=16, use_se=True,
                      model_name="regnety008", **kwargs)


def regnety016(**kwargs):
    """
    RegNetY-1.6GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=48, channels_slope=20.71, channels_mult=2.65, depth=27, groups=24, use_se=True,
                      model_name="regnety016", **kwargs)


def regnety032(**kwargs):
    """
    RegNetY-3.2GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=80, channels_slope=42.63, channels_mult=2.66, depth=21, groups=24, use_se=True,
                      model_name="regnety032", **kwargs)


def regnety040(**kwargs):
    """
    RegNetY-4.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=96, channels_slope=31.41, channels_mult=2.24, depth=22, groups=64, use_se=True,
                      model_name="regnety040", **kwargs)


def regnety064(**kwargs):
    """
    RegNetY-6.4GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=112, channels_slope=33.22, channels_mult=2.27, depth=25, groups=72, use_se=True,
                      model_name="regnety064", **kwargs)


def regnety080(**kwargs):
    """
    RegNetY-8.0GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=192, channels_slope=76.82, channels_mult=2.19, depth=17, groups=56, use_se=True,
                      model_name="regnety080", **kwargs)


def regnety120(**kwargs):
    """
    RegNetY-12GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=168, channels_slope=73.36, channels_mult=2.37, depth=19, groups=112, use_se=True,
                      model_name="regnety120", **kwargs)


def regnety160(**kwargs):
    """
    RegNetY-16GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=200, channels_slope=106.23, channels_mult=2.48, depth=18, groups=112, use_se=True,
                      model_name="regnety160", **kwargs)


def regnety320(**kwargs):
    """
    RegNetY-32GF model from 'Designing Network Design Spaces,' https://arxiv.org/abs/2003.13678.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_regnet(channels_init=232, channels_slope=115.89, channels_mult=2.53, depth=20, groups=232, use_se=True,
                      model_name="regnety320", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        regnetx002,
        regnetx004,
        regnetx006,
        regnetx008,
        regnetx016,
        regnetx032,
        regnetx040,
        regnetx064,
        regnetx080,
        regnetx120,
        regnetx160,
        regnetx320,
        regnety002,
        regnety004,
        regnety006,
        regnety008,
        regnety016,
        regnety032,
        regnety040,
        regnety064,
        regnety080,
        regnety120,
        regnety160,
        regnety320,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != regnetx002 or weight_count == 2684792)
        assert (model != regnetx004 or weight_count == 5157512)
        assert (model != regnetx006 or weight_count == 6196040)
        assert (model != regnetx008 or weight_count == 7259656)
        assert (model != regnetx016 or weight_count == 9190136)
        assert (model != regnetx032 or weight_count == 15296552)
        assert (model != regnetx040 or weight_count == 22118248)
        assert (model != regnetx064 or weight_count == 26209256)
        assert (model != regnetx080 or weight_count == 39572648)
        assert (model != regnetx120 or weight_count == 46106056)
        assert (model != regnetx160 or weight_count == 54278536)
        assert (model != regnetx320 or weight_count == 107811560)
        assert (model != regnety002 or weight_count == 3162996)
        assert (model != regnety004 or weight_count == 4344144)
        assert (model != regnety006 or weight_count == 6055160)
        assert (model != regnety008 or weight_count == 6263168)
        assert (model != regnety016 or weight_count == 11202430)
        assert (model != regnety032 or weight_count == 19436338)
        assert (model != regnety040 or weight_count == 20646656)
        assert (model != regnety064 or weight_count == 30583252)
        assert (model != regnety080 or weight_count == 39180068)
        assert (model != regnety120 or weight_count == 51822544)
        assert (model != regnety160 or weight_count == 83590140)
        assert (model != regnety320 or weight_count == 145046770)

        batch = 14
        size = 224
        x = torch.randn(batch, 3, size, size)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (batch, 1000))


if __name__ == "__main__":
    _test()
