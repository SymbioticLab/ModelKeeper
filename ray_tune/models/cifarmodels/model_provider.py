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

__all__ = ['get_model']


_models = {
    "DenseNet121": DenseNet121, 
    "DenseNet161": DenseNet161, 
    "DenseNet169": DenseNet169, 
    "DenseNet201": DenseNet201, 
    "DPN26": DPN26, 
    "DPN68": DPN68, 
    "DPN92": DPN92, 
    "DPN98": DPN98, 
    "DPN107": DPN107, 
    "DLA": DLA, 
    "SimpleDLA": SimpleDLA, 
    "PreActResNet18": PreActResNet18, 
    "PreActResNet34": PreActResNet34, 
    "PreActResNet50": PreActResNet50, 
    "PreActResNet101": PreActResNet101, 
    "PreActResNet152": PreActResNet152, 
    "ResNet18": ResNet18, 
    "ResNet34": ResNet34, 
    "ResNet50": ResNet50, 
    "ResNet101": ResNet101, 
    "ResNet152": ResNet152, 
    "ResNeXt29_2x64d": ResNeXt29_2x64d, 
    "ResNeXt29_4x64d": ResNeXt29_4x64d, 
    "ResNeXt29_8x64d": ResNeXt29_8x64d, 
    "ResNeXt29_32x4d": ResNeXt29_32x4d, 
    "se_resnet20": se_resnet20, 
    "se_resnet32": se_resnet32, 
    "se_resnet56": se_resnet56, 
    "se_preactresnet20": se_preactresnet20, 
    "se_preactresnet32": se_preactresnet32, 
    "se_preactresnet56": se_preactresnet56, 
    "ShuffleNetG2": ShuffleNetG2, 
    "ShuffleNetG3": ShuffleNetG3, 
    "ShuffleNetV2": ShuffleNetV2, 
    "MobileNetV2": MobileNetV2, 
    "MobileNetV3": MobileNetV3, 
    "VGG": VGG, 
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

