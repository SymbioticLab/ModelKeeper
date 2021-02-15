import torch as th
import numpy as np
import numpy
import copy

def get_mapping_index(layer, new_width, split_max=False):
    """Generate the unit index to replicate"""

    old_width = layer.weight.size(0)

    if split_max:
        replicate_unit = th.topk((th.dot(th.ones(old_width), layer.weight))+layer.bias, k=new_width-old_width)
    else:
        replicate_unit = th.randint(0, old_width, size=(new_width-old_width,))

    return replicate_unit.tolist()

def bn_identity(dim, width):
    if dim == 4:
        bnorm = th.nn.BatchNorm2d(width)
    elif dim == 5:
        bnorm = th.nn.BatchNorm3d(width)
    else:
        bnorm = th.nn.BatchNorm1d(width)

    bnorm.weight.data.fill_(1)
    bnorm.bias.data.fill_(0)
    bnorm.running_mean.fill_(0)
    bnorm.running_var.fill_(1)

    return bnorm

def bn_wider(bnorm, split_index, new_width):
    """Widen the BN layer"""
    old_width = bnorm.running_mean.size(0)

    nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
    nrunning_var = bnorm.running_var.clone().resize_(new_width)

    # [Fan comment] narrow again since resize may not do the multi-dimension?
    nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
    nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)

    if bnorm.affine:
        nweight = bnorm.weight.data.clone().resize_(new_width)
        nbias = bnorm.bias.data.clone().resize_(new_width)

        nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
        nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

    """Handle new units"""
    for i in range(old_width, new_width):
        idx = split_index[i-old_width]
        nrunning_mean[i] = bnorm.running_mean[idx]
        nrunning_var[i] = bnorm.running_var[idx]

        if bnorm.affine:
            nweight[i] = bnorm.weight.data[idx]
            nbias[i] = bnorm.bias.data[idx]

    bnorm.running_var = nrunning_var
    bnorm.running_mean = nrunning_mean
    if bnorm.affine:
        bnorm.weight.data = nweight
        bnorm.bias.data = nbias
    bnorm.num_features = new_width

    return bnorm

def wider(parent, child, new_width, bnorm=None, out_size=None, mapping_index=None, noise_var=5e-2, weight_norm=True):
    """
    Convert m1 layer to its wider version by adapthing next weight layer and
    possible batch norm layer in btw.
    Args:
        layer1 - module to be wider
        layer2 - follwing module to be adapted to m1
        new_width - new width for m1.
        bn (optional) - batch norm layer, if there is btw m1 and m2
        out_size (list, optional) - necessary for m1 == conv3d and m2 == linear. It
            is 3rd dim size of the output feature map of m1. Used to compute
            the matching Linear layer size
        mapping_index (list, optional) - how to replicate existing units
        noise (bool, True) - add a slight noise to break symmetry btw weights.
        weight_norm (optional, True) - If True, weights are normalized before
            transfering.
    """

    """
    Support operations:
    1. More units/kernels
    2. Larger kernel size by padding zeros in resizing
    """
    m1, m2 = copy.deepcopy(parent), copy.deepcopy(child)

    w1 = m1.weight.data
    w2 = m2.weight.data
    b1 = m1.bias.data

    if "Conv" in m1.__class__.__name__ or "Linear" in m1.__class__.__name__:
        # Convert Linear layers to Conv if linear layer follows target layer
        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            assert w2.size(1) % w1.size(0) == 0, "Linear units need to be multiple"
            if w1.dim() == 4:
                factor = int(np.sqrt(w2.size(1) // w1.size(0)))
                w2 = w2.view(w2.size(0), w2.size(1)//factor**2, factor, factor)
            elif w1.dim() == 5:
                assert out_size is not None,\
                       "For conv3d -> linear out_size is necessary"
                factor = out_size[0] * out_size[1] * out_size[2]
                w2 = w2.view(w2.size(0), w2.size(1)//factor, out_size[0],
                             out_size[1], out_size[2])
        else:
            assert w1.size(0) == w2.size(1), "Module weights are not compatible"
        assert new_width > w1.size(0), "New size should be larger"

        nw1 = th.zeros([new_width, w1.size(1)] + list(w1.size())[2:])
        nw2 = th.zeros([w2.size(0), new_width] + list(w2.size())[2:])
        nb1 = th.zeros(new_width)

        w2 = w2.transpose(0, 1)
        nw2 = nw2.transpose(0, 1)

        old_width = w1.size(0)
        nw1.narrow(0, 0, old_width).copy_(w1)
        nw2.narrow(0, 0, old_width).copy_(w2)
        nb1.narrow(0, 0, old_width).copy_(b1)

        # replicate weights randomly, instead of padding zero simply, as the latter is too sparse
        split_index = mapping_index if mapping_index else get_mapping_index(m1, new_width)
        tracking = dict()
        for i in range(old_width, new_width):
            idx = split_index[i-old_width]
            try:
                tracking[idx].append(i)
            except:
                tracking[idx] = [idx] # since we already have this weight before
                tracking[idx].append(i)

            nw1.select(0, i).copy_(w1.select(0, idx).clone())
            nw2.select(0, i).copy_(w2.select(0, idx).clone())
            nb1[i] = b1[idx]

        # as copy units multiple times
        for idx, d in tracking.items():
            for item in d:
                nw2[item].div_(len(d))

        w2.transpose_(0, 1)
        nw2.transpose_(0, 1)

        m1.out_channels = new_width
        m2.in_channels = new_width

        # add noise to break symmetry if we did not modify the parent layer before
        if noise_var and mapping_index is None:
            noise = np.random.normal(scale=noise_var * nw1.std().item(), size=list(nw1.size()))
            nw1 += th.FloatTensor(noise).type_as(nw1)

        m1.weight.data = nw1
        m1.bias.data = nb1

        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            if w1.dim() == 4:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor**2)
                m2.in_features = new_width*factor**2
            elif w2.dim() == 5:
                m2.weight.data = nw2.view(m2.weight.size(0), new_width*factor)
                m2.in_features = new_width*factor
        else:
            m2.weight.data = nw2

        nbnorm = None
        if bnorm is not None:
            nbnorm = bn_wider(copy.deepcopy(bnorm), split_index, new_width) 

        return m1, m2, nbnorm

# TODO: Consider adding noise to new layer as wider operator.
def deeper(m, nonlin, bnorm_flag=False, weight_norm=True, noise_var=5e-2):
    """
    Deeper operator adding a new layer on topf of the given layer.
    Args:
        m (module) - module to add a new layer onto.
        nonlin (module) - non-linearity to be used for the new layer.
        bnorm_flag (bool, False) - whether add a batch normalization btw.
        weight_norm (bool, True) - if True, normalize weights of m before
            adding a new layer.
        noise (bool, True) - if True, add noise to the new layer weights.
    """
    new_layer_type = m.__class__.__name__

    if "Linear" in m.__class__.__name__:
        m2 = th.nn.Linear(m.out_features, m.out_features)
        m2.weight.data.copy_(th.eye(m.out_features))
        m2.bias.data.zero_()

    elif "Conv" in m.__class__.__name__:
        assert m.kernel_size[0] % 2 == 1, "Kernel size needs to be odd"

        if m.weight.dim() == 4:
            pad_h = int((m.kernel_size[0] - 1) / 2)
            # pad_w = pad_h
            m2 = th.nn.Conv2d(m.out_channels, m.out_channels, kernel_size=m.kernel_size, padding=pad_h)
            m2.weight.data.zero_()
            c = m.kernel_size[0] // 2 # center location of the kernel

        elif m.weight.dim() == 5:
            pad_hw = int((m.kernel_size[1] - 1) / 2)  # pad height and width
            pad_d = int((m.kernel_size[0] - 1) / 2)  # pad depth
            m2 = th.nn.Conv3d(m.out_channels, m.out_channels, kernel_size=m.kernel_size, padding=(pad_d, pad_hw, pad_hw))
            c_wh = m.kernel_size[1] // 2
            c_d = m.kernel_size[0] // 2

        restore = False
        if m2.weight.dim() == 2:
            restore = True
            m2.weight.data = m2.weight.data.view(m2.weight.size(0), m2.in_channels, m2.kernel_size[0], m2.kernel_size[0])

        for i in range(0, m.out_channels):
            if m.weight.dim() == 4:
                m2.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, c, 1).narrow(3, c, 1).fill_(1)
            elif m.weight.dim() == 5:
                m2.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, c_d, 1).narrow(3, c_wh, 1).narrow(4, c_wh, 1).fill_(1)

        if noise_var:
            noise = np.random.normal(scale=noise_var * m2.weight.data.std().item(),
                                     size=list(m2.weight.size()))
            m2.weight.data += th.FloatTensor(noise).type_as(m2.weight.data)

        if restore:
            m2.weight.data = m2.weight.data.view(m2.weight.size(0), m2.in_channels, m2.kernel_size[0], m2.kernel_size[0])

        m2.bias.data.zero_()
    else:
        raise RuntimeError("{} Module not supported".format(m.__class__.__name__))

    s = th.nn.Sequential()
    s.add_module('conv', m)
    if bnorm_flag:
        bnorm = bn_identity(m.weight.dim(), m.weight.size(0))
        s.add_module('bnorm', bnorm)
    if nonlin is not None:
        s.add_module('nonlin', nonlin())
    s.add_module('new_'+new_layer_type, m2)

    return s
