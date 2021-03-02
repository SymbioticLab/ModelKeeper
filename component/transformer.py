
import numpy as np

def get_mapping_index(old_width, new_width):
    """Generate the unit index to replicate"""
    return np.random.randint(0, old_width, size=new_width-old_width).tolist()

def paste_slices(tup):
  pos, w, max_w = tup
  wall_min = max(pos, 0)
  wall_max = min(pos+w, max_w)
  block_min = -min(pos, 0)
  block_max = max_w-max(pos+w, max_w)
  block_max = block_max if block_max != 0 else None
  return slice(wall_min, wall_max), slice(block_min, block_max)

def paste(wall, block, loc):
  loc_zip = zip(loc, block.shape, wall.shape)
  wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
  wall[wall_slices] = block[block_slices]

def widen(parent_w, parent_b, child_w, child_b, bnorm=None, mapping_index=None, noise_factor=5e-2):
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

    n_weight = np.zeros_like(child_w)
    n_bias = np.zeros_like(child_b)

    # TODO: figure out top-k important units
    paste(child_w, parent_w, tuple([0] * len(child_w.shape)))
    paste(child_b, parent_b, tuple([0] * len(child_b.shape)))

    # more units/output channels
    old_width, new_width = parent.shape[0], child_w.shape[0]
    if old_width < new_width:
        widen_units = mapping_index if mapping_index is not None else get_mapping_index(old_width, new_width)

        for i in range(old_width, new_width):
            idx = widen_units[i-old_width]
            n_weight[i] = parent_w[idx]
            n_bias[i] = parent_b[idx]

    noise_w = np.random.normal(scale=noise_factor*n_weight.std(), size=list(n_weight.shape))
    n_weight += noise_w

    return n_weight, n_bias, mapping_index

def widen_child(weight, mapping_index, noise_factor=0):
    n_weight = weight.copy()
    if len(mapping_index) > 0:
        n_weight = n_weight.transpose(0, 1)
        tracking = dict()

        for i in range(old_width, new_width):
            idx = widen_units[i-old_width]

            if idx not in tracking:
                tracking[idx] = [idx]
            tracking[idx].append(i)

            n_weight[i] = weight[idx]

        for idx, d in tracking.items():
            for item in d:
                n_weight[item] /= len(d)

        n_weight = n_weight.transpose(0, 1)
        noise_w = np.random.normal(scale=noise_factor*n_weight.std(), size=list(n_weight.shape))
        n_weight += noise_w

    return n_weight

