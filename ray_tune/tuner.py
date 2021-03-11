from nas_201_api import NASBench201API as API
from models import get_cell_based_tiny_net


import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import torchvision.models as models

import logging

log_path = '/gpfs/gpfs0/groups/chowdhury/dywsjtu/repos/NetTransformer/AutoDL-Projects/lib/logging'
with open(log_path, 'w') as fout:
    pass

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(log_path, mode='a'),
                    logging.StreamHandler()
                ])

def train(model, optimizer, criterion, train_loader, device=torch.device("cpu")):
    """
    Model training function
    Parameters
    ---------
    model : Pytorch model
        Model instantiated
    optimizer : Pytorch optimizer
        Optimization algorithm defined
    criterion : Pytorch loss function
        Loss function defined
    train_loader : Pytorch dataloader
        Contains training data
    device : Pytorch device
        cpu or cuda
    Returns
    ---------
    Training for one epoch
    """
    model.to(device).train()
    for data, target in train_loader:
        data, target = Variable(data.to(device)), Variable(target.to(device))
        optimizer.zero_grad()
        _, output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def eval(model, criterion, data_loader, device=torch.device("cpu")):
    """
    Model evaluation function
    Parameters
    ---------
    model : Pytorch model
        Model instantiated
    criterion : Pytorch loss function
        Loss function defined
    data_loader : Pytorch dataloader
        Contains evaluation data
    device : Pytorch device
        cpu or cuda
    Returns
    ---------
    accuracy, loss : tuple
        Accuracy and loss of the evaluated model
    
    """
    model.to(device).eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = Variable(data.to(device)), Variable(target.to(device))
            _, output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, target)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Cifar10 Example")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging status')
    parser.add_argument('--noise', type=float, default=5e-2,
                        help='noise or no noise 0-1')
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--dataset', type=str, default='/gpfs/gpfs0/groups/chowdhury/fanlai/dataset')
    parser.add_argument('--meta', type=str, default='/gpfs/gpfs0/groups/chowdhury/dywsjtu/NAS-Bench-201-v1_1-096897.pth')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
    num_of_class = 10

    if args.data == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset, train=True, download=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset, train=False, download=True, transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)



    api = API(args.meta, verbose=False)

    config = api.get_net_config(123, 'cifar10') 
    print(config)
    config = api.get_net_config(124, 'cifar10')
    print(config)
    model = get_cell_based_tiny_net(config) 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = nn.NLLLoss()

    # if args.cuda and torch.cuda.is_available():
    #     print("Use cuda")
    #     train(model, optimizer, criterion, train_loader, torch.device("cuda"))
    #     eval(model, criterion, test_loader, torch.device("cuda"))

    # else:
    #     print("Use cpu")
    #     train(model, optimizer, criterion, train_loader, torch.device("cpu"))
    #     eval(model, criterion, test_loader, torch.device("cpu"))
