import torch as th
import numpy as np
import numpy
import copy
import warnings

def get_mapping_index(layer, new_width, split_max=False):
    """Generate the unit index to replicate"""

    old_width = layer.weight.size(0)

    if split_max:
        replicate_unit = th.topk((th.dot(th.ones(old_width), layer.weight))+layer.bias, k=new_width-old_width)
        return replicate_unit.tolist()
    else:
        replicate_unit = []
        for i in range(old_width, new_width):
            replicate_unit.append(np.random.randint(0, i))

        return replicate_unit

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

    bnorm.running_mean.resize_(new_width)
    bnorm.running_var.resize_(new_width)

    if bnorm.affine:
        bnorm.weight.data.resize_(new_width)
        bnorm.bias.data.resize_(new_width)

    """Handle new units"""
    for i in range(old_width, new_width):
        idx = split_index[i-old_width]
        bnorm.running_mean[i] = bnorm.running_mean[idx].clone()
        bnorm.running_var[i] = bnorm.running_var[idx].clone()

        if bnorm.affine:
            bnorm.weight[i] = bnorm.weight[idx].clone()
            bnorm.bias[i] = bnorm.bias[idx].clone()
            
    bnorm.num_features = new_width

    return bnorm

def wider(parent, child, new_width, bnorm=None, template_layer=None, out_size=None, mapping_index=None, noise_var=5e-2):
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

    if template_layer is None: 
        template_layer = m1

    w1 = m1.weight.data
    w2 = m2.weight.data
    b1 = m1.bias.data

    new_layer_type = template_layer.__class__.__name__

    if "Conv" in new_layer_type or "Linear" in template_layer.__class__.__name__:
        # Convert Linear layers to Conv if linear layer follows target layer
        if "Conv" in new_layer_type and "Linear" in m2.__class__.__name__:
            assert w2.size(1) % template_layer.weight.size(0) == 0, "Linear units need to be multiple"
            if template_layer.weight.dim() == 4:
                factor = int(np.sqrt(w2.size(1) // template_layer.weight.size(0)))
                w2 = w2.view(w2.size(0), w2.size(1)//factor**2, factor, factor)
            elif template_layer.weight.dim() == 5:
                assert out_size is not None,\
                       "For conv3d -> linear out_size is necessary"
                factor = out_size[0] * out_size[1] * out_size[2]
                w2 = w2.view(w2.size(0), w2.size(1)//factor, out_size[0],
                             out_size[1], out_size[2])
        else:
            assert w1.size(0) == w2.size(1), "Module weights are not compatible"
        assert new_width > w1.size(0), "New size should be larger"

        new_width = template_layer.weight.size(0)

        nw1 = th.zeros_like(template_layer.weight)
        nw2 = th.zeros([w2.size(0), new_width] + list(w2.size())[2:])
        nb1 = th.zeros(new_width)

        w2 = w2.transpose(0, 1)
        nw2 = nw2.transpose(0, 1)
        old_width = w1.size(0)

        nw1.narrow(0, 0, old_width).copy_(w1)
        nw2.narrow(0, 0, old_width).copy_(w2)
        nb1.narrow(0, 0, old_width).copy_(b1)

        # replicate weights randomly, instead of padding zero simply, as the latter is too sparse
        split_index = mapping_index
        if split_index is None:
            split_index = get_mapping_index(m1, new_width)

        # add noise to break symmetry if we did not modify the parent layer before
        for i in range(old_width, new_width):
            idx = split_index[i-old_width]
            nw1[i] = nw1[idx].clone()

            # halve and then add noise
            split_weight = nw2[idx].clone() * 0.5
            noise_w = np.random.normal(scale=noise_var * nw2[idx].std().item(), size=list(nw2[idx].size()))
            nw2[i] = split_weight + th.FloatTensor(noise_w)
            nw2[idx] = split_weight - th.FloatTensor(noise_w)

            # bias layer
            noise_bn = np.random.normal(scale=abs(noise_var * nb1[idx].item()))
            nb1[i] = nb1[idx].clone()
            nb1[idx] -= noise_bn
            nb1[i] += noise_bn
            
        w2.transpose_(0, 1)
        nw2.transpose_(0, 1)

        m2.in_channels = new_width

        template_layer.weight.data = nw1
        template_layer.bias.data = nb1

        if "Conv" in new_layer_type and "Linear" in m2.__class__.__name__:
            if w2.dim() == 4:
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

        return template_layer, m2, nbnorm

# Leave different kernel size to the widen operation
def deeper(m, nonlin, bnorm_flag=False, noise_var=5e-2):
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

    if "Linear" in new_layer_type:
        m2 = th.nn.Linear(m.out_features, m.out_features)
        m2.weight.data.zero_()
        m2.bias.data.zero_()
        m2.weight.data.copy_(th.eye(m.out_features))

    elif "Conv" in new_layer_type:
        assert m.kernel_size[0] % 2 == 1, "Kernel size needs to be odd"

        if m.weight.dim() == 4:
            pad_h = m.kernel_size[0] // 2
            pad_w = m.kernel_size[1] // 2
            m2 = th.nn.Conv2d(m.out_channels, m.out_channels, kernel_size=m.kernel_size, padding=(pad_h, pad_w))
            c_d = m.kernel_size[0] // 2 # center location of the kernel
            c_wh = m.kernel_size[1] // 2

        elif m.weight.dim() == 5:
            pad_hw = m.kernel_size[1] // 2  # pad height and width
            pad_d = m.kernel_size[0] // 2  # pad depth
            m2 = th.nn.Conv3d(m.out_channels, m.out_channels, kernel_size=m.kernel_size, padding=(pad_d, pad_hw, pad_hw))
            c_wh = m.kernel_size[1] // 2
            c_d = m.kernel_size[0] // 2

        m2.weight.data.zero_()
        m2.bias.data.zero_()

        restore = False
        if m2.weight.dim() == 2:
            restore = True
            m2.weight.data = m2.weight.data.view(m2.weight.size(0), m2.in_channels, m2.kernel_size[0], m2.kernel_size[0])

        for i in range(0, m.out_channels):
            if m.weight.dim() == 4:
                m2.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, c_d, 1).narrow(3, c_wh, 1).fill_(1)
            elif m.weight.dim() == 5:
                m2.weight.data.narrow(0, i, 1).narrow(1, i, 1).narrow(2, c_d, 1).narrow(3, c_wh, 1).narrow(4, c_wh, 1).fill_(1)

        if noise_var:
            # no need std here since it is eye
            w_noise = np.random.normal(scale=noise_var, size=list(m2.weight.size()))
            m2.weight.data += th.FloatTensor(w_noise).type_as(m2.weight.data)

            bn_noise = np.random.normal(scale=noise_var, size=list(m2.bias.size()))
            m2.bias.data += th.FloatTensor(bn_noise).type_as(m2.bias.data)

        if restore:
            m2.weight.data = m2.weight.data.view(m2.weight.size(0), m2.in_channels, m2.kernel_size[0], m2.kernel_size[0])

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
