import math
import os
import pickle

import torch


def load_model(file):
    with open(file, 'rb') as fin:
        _, model = pickle.load(fin), pickle.load(fin)
    model = model.to(device='cpu')
    return model

def diff_model(m1_p, m2_p):
    m1, m2 = load_model(m1_p), load_model(m2_p)
    grad_norms = []
    grad_vars = []
    p1_norms = []

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1.requires_grad:
            diff = p1 - p2
            grad_norms.append(diff.norm(2).item())
            grad_vars.append(diff.var(unbiased=False).norm(2).item()/abs(diff.mean().item()))
            p1_norms.append(p1.norm(2))
        else:
            grad_norms.append(None)
            grad_vars.append(None)
            p1_norms.append(None)

    return grad_norms, grad_vars, p1_norms

def get_grad_norm(file):
    with open(file, 'rb') as f:
        gradients, model = pickle.load(f), pickle.load(f)

    gradient_norm = []
    gradient_var = []
    for g, p in zip(gradients, model.parameters()):
        if len(g) > 0:
            gradient_norm.append(g.norm(2))
            gradient_var.append(g.var(unbiased=False).norm(2))
        else:
            gradient_norm.append(None)
            gradient_var.append(None)

    return gradient_norm, gradient_var

def get_current_lr(init_lr, min_lr, step, max_t):
    return (min_lr+0.5*(init_lr-min_lr)*(1+math.cos(math.pi * step/ max_t)))

mypath = "/users/fanlai/experiment/ModelKeeper/scripts/motivation/warm_cifar10"

def enumerate_diff(path):
    grad_norms = []
    grad_vars = []
    grad_lr = []
    p1_norms = []
    lr = 0.01 if 'cold' in path else 0.003
    min_lr = 1e-3
    max_t = 125

    for i in range(0, 125):
        norm, var, p1_norm = diff_model(os.path.join(path, f"ResNet101_cifar10_{i}_0.pkl"), os.path.join(path, f"ResNet101_cifar10_{i}_195.pkl"))
        grad_norms.append(norm)
        grad_vars.append(var)
        grad_lr.append(get_current_lr(lr, min_lr, i, max_t))
        p1_norms.append(p1_norm)
        print(f"Done {i}...")

    with open(path.split('/')[-1]+'_grad.pkl', 'wb') as fout:
        pickle.dump(grad_norms, fout)
        pickle.dump(grad_vars, fout)
        pickle.dump(p1_norms, fout)
        pickle.dump(grad_lr, fout)

    #return grad_result

def cal_norm_var(path):
    gradient_norms = []
    gradient_vars = []

    for i in range(55):
        for k in [0, 195]:
            norm, var = get_grad_norm(os.path.join(path, f"ResNet101_cifar10_{i}_{k}.pkl"))
            gradient_norms.append(norm)
            gradient_vars.append(var)

    with open(path.split('/')[-1]+'_grad.pkl', 'wb') as fout:
        pickle.dump(gradient_norms, fout)
        pickle.dump(gradient_vars, fout)

#cal_norm_var(mypath)
enumerate_diff(mypath)

#print(diff_model(os.path.join(path, "ResNet101_cifar10_2.pkl"), os.path.join(path, "ResNet101_cifar10_118.pkl")))
