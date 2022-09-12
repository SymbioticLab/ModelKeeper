import numpy as np
import torch


def get_mapping_index(old_width, new_width):
    """Generate the unit index to replicate"""
    return list([x % old_width for x in range(new_width - old_width)])


def paste_slices(tup):
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos + w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w - max(pos + w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]


def widen(
        parent_w,
        parent_b,
        child_w,
        child_b,
        bnorm=None,
        mapping_index=None,
        noise_factor=5e-2):
    """
    Convert m1 layer to its wider version by adapthing next weight layer and
    possible batch norm layer in btw.
    Args:
        layer1 - module to be wider
        layer2 - follwing module to be adapted to m1
        bn (optional) - batch norm layer, if there is btw m1 and m2
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
    n_weight = n_bias = None

    if parent_w is not None:
        n_weight = np.zeros_like(child_w)
        paste(n_weight, parent_w, tuple([0] * len(n_weight.shape)))
        # more units/output channels
        old_width, new_width = parent_w.shape[0], child_w.shape[0]

    # TODO: figure out top-k important units
    if parent_b is not None and child_b is not None:
        n_bias = np.zeros_like(child_b)
        paste(n_bias, parent_b, tuple([0] * len(n_bias.shape)))
        old_width, new_width = parent_b.shape[0], child_b.shape[0]

    widen_units = []

    if old_width < new_width:
        widen_units = mapping_index if mapping_index is not None else get_mapping_index(
            old_width, new_width)

        for i in range(old_width, new_width):
            idx = widen_units[i - old_width]
            if n_weight is not None:
                n_weight[i] = n_weight[idx].copy() + np.random.normal(scale=noise_factor * \
                                                 n_weight[idx].std(), size=list(n_weight[idx].shape))

            if n_bias is not None:
                n_bias[i] = n_bias[idx].copy() + np.random.normal(scale=noise_factor * \
                                             n_bias[idx].std(), size=list(n_bias[idx].shape))

    return n_weight, n_bias, widen_units, new_width


def widen_child(weight, bias, mapping_index, new_width):
    if len(mapping_index) > 0:
        tracking = dict()
        n_weight = torch.from_numpy(weight)
        n_bias = torch.from_numpy(bias) if bias is not None else None

        is_bn = len(n_weight.shape) == 1

        if not is_bn:
            n_weight.transpose_(0, 1)

        # new_width = n_weight.shape[0]
        old_width = new_width - len(mapping_index)

        # Parent nodes already add some noise, so no need to add again
        for i in range(old_width, new_width):
            idx = mapping_index[i - old_width]
            if idx not in tracking:
                tracking[idx] = [idx]
            tracking[idx].append(i)

            n_weight[i] = n_weight[idx].clone()
            if n_bias is not None:
                n_bias[i] = n_bias[idx].clone()

        if not is_bn:
            for idx, d in tracking.items():
                for item in d:
                    n_weight[item] /= float(len(d))

                    if n_bias is not None:
                        n_bias[item] /= float(len(d))

            n_weight.transpose_(0, 1)

        n_weight = n_weight.numpy()
        #n_weight += np.random.normal(scale=noise_factor*n_weight.std(), size=list(n_weight.shape))
        return n_weight, n_bias
    else:
        return weight, bias


def deepen(weight, noise_factor=5e-2):
    """Build an identity layer"""
    n_bias = np.zeros(weight.shape[0], dtype=weight.dtype)
    n_weight = None

    # 2-D Linear layers
    if len(weight.shape) == 1:  # BN layer
        n_weight = np.ones(weight.shape[0])
    elif len(weight.shape) == 2:
        n_weight = np.matrix(np.eye(weight.shape[0], dtype=weight.dtype))
    elif len(weight.shape) > 2:
        c_d, c_wh = weight.shape[2] // 2, weight.shape[3] // 2

        n_weight = torch.zeros(weight.shape)
        for i in range(n_weight.shape[1]):
            if n_weight.dim() == 4:
                n_weight.narrow(
                    0,
                    i,
                    1).narrow(
                    1,
                    i,
                    1).narrow(
                    2,
                    c_d,
                    1).narrow(
                    3,
                    c_d,
                    1).fill_(1)
            elif n_weight.dim() == 5:
                n_weight.narrow(
                    0,
                    i,
                    1).narrow(
                    1,
                    i,
                    1).narrow(
                    2,
                    c_d,
                    1).narrow(
                    3,
                    c_wh,
                    1).narrow(
                    4,
                    c_wh,
                    1).fill_(1)

        n_weight = n_weight.numpy()

    # add noise
    if n_weight is not None:
        n_weight += np.random.normal(scale=noise_factor,
                                     size=list(n_weight.shape))
        n_bias += np.random.normal(scale=noise_factor, size=list(n_bias.shape))
        return n_weight, n_bias

    return weight, n_bias
