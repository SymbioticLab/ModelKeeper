from .vgg import *
from .dpn import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenetv3 import *
from .mobilenetv2 import *
from .efficientnet import *
from .dla_simple import *
from .dla import *
from .preactresnet import *
from .stochasticdepth import *
from .preact_resnet import *

__all__ = ['get_model']


_models = {
    "DLA": DLA,
    "DPN107": DPN107,
    "DPN26": DPN26,
    "DPN68": DPN68,
    "DPN92": DPN92,
    "DPN98": DPN98,
    "DenseNet121": DenseNet121,
    "DenseNet161": DenseNet161,
    "DenseNet169": DenseNet169,
    "DenseNet201": DenseNet201,
    "MobileNetV2": MobileNetV2,
    "MobileNetV3": MobileNetV3,
    "PreActResNet101": PreActResNet101,
    "PreActResNet152": PreActResNet152,
    "PreActResNet18": PreActResNet18,
    "PreActResNet34": PreActResNet34,
    "PreActResNet50": PreActResNet50,
    "ResNeXt29_2x64d": ResNeXt29_2x64d,
    "ResNeXt29_32x4d": ResNeXt29_32x4d,
    "ResNeXt29_4x64d": ResNeXt29_4x64d,
    "ResNeXt29_8x64d": ResNeXt29_8x64d,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "ResNet18": ResNet18,
    "ResNet34": ResNet34,
    "ResNet50": ResNet50,
    "ShuffleNetG2": ShuffleNetG2,
    "ShuffleNetG3": ShuffleNetG3,
    "ShuffleNetV2": ShuffleNetV2,
    "SimpleDLA": SimpleDLA,
    "VGG": VGG,
    "preactresnet101": preactresnet101,
    "preactresnet152": preactresnet152,
    "preactresnet18": preactresnet18,
    "preactresnet34": preactresnet34,
    "preactresnet50": preactresnet50,
    "seresnet101": seresnet101,
    "seresnet152": seresnet152,
    "seresnet18": seresnet18,
    "seresnet34": seresnet34,
    "seresnet50": seresnet50,
    "stochastic_depth_resnet101": stochastic_depth_resnet101,
    "stochastic_depth_resnet152": stochastic_depth_resnet152,
    "stochastic_depth_resnet18": stochastic_depth_resnet18,
    "stochastic_depth_resnet34": stochastic_depth_resnet34,
    "stochastic_depth_resnet50": stochastic_depth_resnet50,
}


def get_cv_model(name, **kwargs):
    """
    Get supported model.

    Parameters:
    ----------
    name : str
        Name of model.

    Returns:
    -------
    Module
        Resulted model.
    """
    if name not in _models:
        raise ValueError("Unsupported model: {}".format(name))
    net = _models[name](**kwargs)
    return net


