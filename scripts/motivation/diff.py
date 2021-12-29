import torch, os, pickle

def load_model(file):
    with open(file, 'rb') as fin:
        _, _, model = pickle.load(fin), pickle.load(fin), pickle.load(fin)
    model = model.to(device='cpu')
    return model 

def diff_model(m1_p, m2_p):
    m1, m2 = load_model(m1_p), load_model(m2_p)
    grad_norms = []
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        grad_norms.append((p1-p2).norm(2).item())
    return grad_norms

mypath = "/users/fanlai/experiment/ModelKeeper/scripts/motivation/arxiv/cold_cifar10"

def enumerate_diff(path):
    grad_result = []
    for i in range(2, 116, 2):
        grad_result.append(diff_model(os.path.join(path, f"ResNet101_cifar10_{i}.pkl"), os.path.join(path, f"ResNet101_cifar10_{i+2}.pkl")))
        print(f"Done {i}...")
    with open(mypath.split('/')[-1]+'_grad.pkl', 'wb') as fout:
        pickle.dump(grad_result, fout)

    return grad_result

enumerate_diff(mypath)

#print(diff_model(os.path.join(path, "ResNet101_cifar10_2.pkl"), os.path.join(path, "ResNet101_cifar10_118.pkl")))


