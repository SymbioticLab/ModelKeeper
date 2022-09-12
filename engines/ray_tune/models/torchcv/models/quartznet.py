"""
    QuartzNet for ASR, implemented in PyTorch.
    Original paper: 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel Separable Convolutions,'
    https://arxiv.org/abs/1910.10261.
"""

__all__ = ['quartznet5x5_en_ls', 'quartznet15x5_en', 'quartznet15x5_en_nr', 'quartznet15x5_fr', 'quartznet15x5_de',
           'quartznet15x5_it', 'quartznet15x5_es', 'quartznet15x5_ca', 'quartznet15x5_pl', 'quartznet15x5_ru',
           'quartznet15x5_ru34']

from .jasper import get_jasper


def quartznet5x5_en_ls(num_classes=29, **kwargs):
    """
    QuartzNet 5x5 model for English language (trained on LibriSpeech dataset) from 'QuartzNet: Deep Automatic Speech
    Recognition with 1D Time-Channel Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                  't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]
    return get_jasper(num_classes=num_classes, version=("quartznet", "5x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet5x5_en_ls", **kwargs)


def quartznet15x5_en(num_classes=29, **kwargs):
    """
    QuartzNet 15x5 model for English language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                  't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_en", **kwargs)


def quartznet15x5_en_nr(num_classes=29, **kwargs):
    """
    QuartzNet 15x5 model for English language (with presence of noise) from 'QuartzNet: Deep Automatic Speech
    Recognition with 1D Time-Channel Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                  't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_en_nr", **kwargs)


def quartznet15x5_fr(num_classes=43, **kwargs):
    """
    QuartzNet 15x5 model for French language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 43
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                  't', 'u', 'v', 'w', 'x', 'y', 'z', "'", 'ç', 'é', 'â', 'ê', 'î', 'ô', 'û', 'à', 'è', 'ù', 'ë', 'ï',
                  'ü', 'ÿ']
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_fr", **kwargs)


def quartznet15x5_de(num_classes=32, **kwargs):
    """
    QuartzNet 15x5 model for German language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 32
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                  't', 'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'ü', 'ß']
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_de", **kwargs)


def quartznet15x5_it(num_classes=39, **kwargs):
    """
    QuartzNet 15x5 model for Italian language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 39
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                  't', 'u', 'v', 'w', 'x', 'y', 'z', "'", 'à', 'é', 'è', 'í', 'ì', 'î', 'ó', 'ò', 'ú', 'ù']
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_it", **kwargs)


def quartznet15x5_es(num_classes=36, **kwargs):
    """
    QuartzNet 15x5 model for Spanish language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 36
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                  't', 'u', 'v', 'w', 'x', 'y', 'z', "'", 'á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü']
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_es", **kwargs)


def quartznet15x5_ca(num_classes=39, **kwargs):
    """
    QuartzNet 15x5 model for Spanish language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 39
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                  't', 'u', 'v', 'w', 'x', 'y', 'z', "'", 'à', 'é', 'è', 'í', 'ï', 'ó', 'ò', 'ú', 'ü', 'ŀ']
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_ca", **kwargs)


def quartznet15x5_pl(num_classes=34, **kwargs):
    """
    QuartzNet 15x5 model for Spanish language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 34
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n', 'ń',
                  'o', 'ó', 'p', 'r', 's', 'ś', 't', 'u', 'w', 'y', 'z', 'ź', 'ż']
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_pl", **kwargs)


def quartznet15x5_ru(num_classes=35, **kwargs):
    """
    QuartzNet 15x5 model for Russian language from 'QuartzNet: Deep Automatic Speech Recognition with 1D Time-Channel
    Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 35
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ё', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с',
                  'т', 'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_ru", **kwargs)


def quartznet15x5_ru34(num_classes=34, **kwargs):
    """
    QuartzNet 15x5 model for Russian language (32 graphemes) from 'QuartzNet: Deep Automatic Speech Recognition with 1D
    Time-Channel Separable Convolutions,' https://arxiv.org/abs/1910.10261.

    Parameters:
    ----------
    num_classes : int, default 34
        Number of classification classes (number of graphemes).
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    vocabulary = [' ', 'а', 'б', 'в', 'г', 'д', 'е', 'ж', 'з', 'и', 'й', 'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т',
                  'у', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ъ', 'ы', 'ь', 'э', 'ю', 'я']
    return get_jasper(num_classes=num_classes, version=("quartznet", "15x5"), use_dw=True, vocabulary=vocabulary,
                      model_name="quartznet15x5_ru34", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import numpy as np
    import torch

    pretrained = False
    from_audio = False
    audio_features = 64
    use_cuda = True

    models = [
        quartznet5x5_en_ls,
        quartznet15x5_en,
        quartznet15x5_en_nr,
        quartznet15x5_fr,
        quartznet15x5_de,
        quartznet15x5_it,
        quartznet15x5_es,
        quartznet15x5_ca,
        quartznet15x5_pl,
        quartznet15x5_ru,
        quartznet15x5_ru34,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            from_audio=from_audio,
            pretrained=pretrained)

        if use_cuda:
            net = net.cuda()

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != quartznet5x5_en_ls or weight_count == 6713181)
        assert (model != quartznet15x5_en or weight_count == 18924381)
        assert (model != quartznet15x5_en_nr or weight_count == 18924381)
        assert (model != quartznet15x5_fr or weight_count == 18938731)
        assert (model != quartznet15x5_de or weight_count == 18927456)
        assert (model != quartznet15x5_it or weight_count == 18934631)
        assert (model != quartznet15x5_es or weight_count == 18931556)
        assert (model != quartznet15x5_ca or weight_count == 18934631)
        assert (model != quartznet15x5_pl or weight_count == 18929506)
        assert (model != quartznet15x5_ru or weight_count == 18930531)
        assert (model != quartznet15x5_ru34 or weight_count == 18929506)

        batch = 3
        aud_scale = 640 if from_audio else 1
        seq_len = np.random.randint(150, 250, batch) * aud_scale
        seq_len_max = seq_len.max() + 2
        x_shape = (batch, seq_len_max) if from_audio else (batch, audio_features, seq_len_max)
        x = torch.randn(x_shape)
        x_len = torch.tensor(seq_len, dtype=torch.long, device=x.device)

        if use_cuda:
            x = x.cuda()
            x_len = x_len.cuda()

        y, y_len = net(x, x_len)
        # y.sum().backward()

        assert (tuple(y.size())[:2] == (batch, net.num_classes))
        if from_audio:
            assert (y.size()[2] in range(seq_len_max // aud_scale * 2, seq_len_max // aud_scale * 2 + 9))
        else:
            assert (y.size()[2] in [seq_len_max // 2, seq_len_max // 2 + 1])


if __name__ == "__main__":
    _test()
