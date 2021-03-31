import torch
import numpy as np
import networkx as nx
from itertools import permutations


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args, cuda='cuda'):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.to(cuda)
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def make_graph(recepie):
    G = nx.DiGraph()

    for key in recepie.keys():
        op = recepie[key]['op']
        if key.startswith("h_new_"):
            op = key+":"+op
        G.add_node(key, name=key, op=op)
        for inp in recepie[key]['input']:
            if "h_prev" in inp or inp == "x":
                G.add_node(inp, name=inp, op=inp)
            else:
                G.add_node(inp, name=inp)
            G.add_edge(inp, key)
    return G


def recepie2matrixops(recepie):
    G = make_graph(recepie)
    labels = nx.get_node_attributes(G, "op")
    nodelist_with_ops = np.array(list(labels.items()))
    
    matrix = nx.to_numpy_array(G, nodelist=nodelist_with_ops[:, 0])
    ops = nodelist_with_ops[:, 1]

    return matrix, ops



def graph_edit_distance(matrixops1, matrixops2):
    m1, l1 = matrixops1
    m2, l2 = matrixops2
    
    # Pad
    n1, n2 = m1.shape[0], m2.shape[0]
    max_n = max(n1, n2)
    m1 = np.pad(m1, ((0, max_n - m1.shape[0]), (0, max_n - m1.shape[0])))
    m2 = np.pad(m2, ((0, max_n - m2.shape[0]), (0, max_n - m2.shape[0])))
    l1 = np.pad(l1, (0, max_n - l1.shape[0]), constant_values=None)
    l2 = np.pad(l2, (0, max_n - l2.shape[0]), constant_values=None)
    
    
    d = 100000000
    for p in permutations(range(len(m1))):
        p = list(p)
        d_p = (m1 != m2[p][:, p]).sum() + (l1 != l2[p]).sum()
        d = min(d, d_p)
    return d
