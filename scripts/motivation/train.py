from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
#sys.path.append('../')
#from net2net import *
import copy
import torchvision.models as tormodels
import pickle
import logging
from vgg import *

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO,
                handlers=[
                    #logging.FileHandler(log_path, mode='a'),
                    logging.StreamHandler()
                ])

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--model', type=str, default="vgg19_bn")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epoch', type=int, default=300, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=224, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                    help='learning rate (default: 0.01)') #0.002
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging status')
parser.add_argument('--data', type=str, default='cifar100')
parser.add_argument('--weight_decay', type=float, default=1e-4)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
seed = args.seed

torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}

train_transform = transforms.Compose(
             [transforms.RandomCrop(32, padding=4),
             transforms.Resize(32),
              transforms.RandomHorizontalFlip(),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

test_transform = transforms.Compose(
             [transforms.Resize(32),
              transforms.ToTensor(),
              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

imagnet_categories = 300
data_categories = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000, 'ImageNet120': imagnet_categories}



if args.data == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/users/fanlai/', train=True, download=True, transform=train_transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/users/fanlai/', train=False, download=True, transform=test_transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

elif args.data == 'cifar100':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/users/fanlai/experiment/data', train=True, download=True, transform=train_transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/users/fanlai/experiment/data', train=False, download=True, transform=test_transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

elif args.data == 'imagenet':
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageNet('/users/fanlai/imagenet/', split='train',
                       transform=transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageNet('/users/fanlai/imagenet/', split='val',
                        transform=transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

elif args.data == 'ImageNet120':
    sys.path.insert(0,'../../ray_tune/')
    from ImageNet import ImageNet16

    mean = [x / 255 for x in [122.68, 116.66, 104.01]]
    std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
    lists = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(), transforms.Normalize(mean, std)]
    train_transform = transforms.Compose(lists)
    test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data = ImageNet16('/users/fanlai/imagenet32', True , train_transform, size=32, use_num_of_class_only=imagnet_categories)
    test_data  = ImageNet16('/users/fanlai/imagenet32', False, test_transform, size=32, use_num_of_class_only=imagnet_categories)
    print(len(train_data), len(test_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)


try:
    model = eval(f"{args.model}({data_categories[args.data]})")#
except Exception as e:
    model = tormodels.__dict__[args.model](num_classes=data_categories[args.data])

if args.cuda:
    model.cuda()


device = 'cuda' if args.cuda else 'cpu'

def prefix_warmup(file):
    with open(file, 'rb') as fin:
        epoch = pickle.load(fin)
        opt = pickle.load(fin)
        cmodel = pickle.load(fin)

    # prefix match
    p_len = 0
    for param in model.parameters():
        p_len += 1
    c_len = 0
    for param in cmodel.parameters():
        c_len += 1

    idx = 0
    for c_param, p_param in zip(cmodel.parameters(), model.parameters()):
        if c_param.data.shape == p_param.data.shape:
            p_param.data = c_param.data.clone()
            idx += 1
        else:
            break

    # Update learning rate

    args.lr *= (1. - idx/c_len)
    logging.info(f"Prefix match {idx} layers out of child {p_len}, parent {c_len} layers")


def greedy_prefix_warmup(file):
    with open(file, 'rb') as fin:
        epoch = pickle.load(fin)
        opt = pickle.load(fin)
        cmodel = pickle.load(fin)

    # prefix match
    p_len = 0
    for param in model.parameters():
        #logging.info(param.data.shape)
        p_len += 1
    c_len = 0
    for param in cmodel.parameters():
        #logging.info(param.data.shape)
        c_len += 1

    p_idx = 0
    cnt = 0
    for cparam in cmodel.parameters():
        for idx, param in enumerate(model.parameters()):
            if idx >= p_idx:
                if param.data.shape == cparam.data.shape:
                    param.data = cparam.data.clone()
                    p_idx = idx + 1
                    cnt += 1
                    break

    logging.info(f"Prefix match {cnt} layers out of child {p_len}, parent {c_len} layers")


def modelkeeper(model):
    import sys
    sys.path.insert(0,'../../ray_tune/modelkeeper')

    from matchingopt import ModelKeeper
    from config import modelkeeper_config

    modelkeeper_config.zoo_path = '/mnt/zoo/vgg_zoo'
    mapper = ModelKeeper(modelkeeper_config)
    model = model.to(device='cpu')

    dummy_input = torch.rand(8, 3, 32, 32)
    weights, meta_data = mapper.map_for_model(model, dummy_input)

    for name, p in model.named_parameters():
        if weights is not None:
            temp_data = (torch.from_numpy(weights[name])).data
            assert(temp_data.shape == p.data.shape)
            p.data = temp_data.to(dtype=p.data.dtype)

    print(f"ModelKeeper mapping meta: \n{meta_data}")
    args.lr *= (1.-meta_data['matching_score']+1e-4)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    # lr = args.lr * (0.5 ** (epoch // 20))
    if epoch < 120:
        lr = 0.02
    elif epoch < 160:
        lr = 0.004
    else:
        lr = 0.0008

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss().to(device=device)

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % 1000 == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    logging.info('====Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))


def test():
    model.eval()
    test_loss = 0
    top_1, top_k = 0, 0
    k = 5

    criterion = torch.nn.CrossEntropyLoss().to(device=device)

    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss

            _, maxk = torch.topk(output, k, dim=-1)
            test_labels = target.view(-1, 1)

            top_1 += (test_labels == maxk[:, 0:1]).sum().item()
            top_k += (test_labels == maxk).sum().item()

    test_data_len = len(test_loader.dataset)
    test_loss /= len(test_loader)
    logging.info('Test set: Average loss: {:.4f}, Accuracy (Top-1): {}/{} ({:.3f}%), (Top-{}) {:.3f}%'.format(
        test_loss, top_1, test_data_len, 100. * top_1 / test_data_len, k, 100. * top_k / test_data_len))

    return 100. * top_1 / test_data_len, test_loss

def dump_model(epoch, optimizer, model):
    with open(f"./zoo/{args.model}_{args.data}_{epoch}.pkl", 'wb') as fout:
        pickle.dump(epoch, fout)
        pickle.dump(optimizer, fout)
        pickle.dump(model, fout)

vgg16_match = '/users/fanlai/ModelKeeper/scripts/motivation/zoo/vgg16_bn_cifar100_299.pkl' #70%: 4, 70%:, 85%: 24, 90%:199
vgg11_match = '/users/fanlai/ModelKeeper/scripts/motivation/zoo/vgg11_bn_cifar100_299.pkl'
vgg13_match = '/users/fanlai/ModelKeeper/scripts/motivation/zoo/vgg13_bn_cifar100_299.pkl'


#prefix_warmup(vgg16_match)
#modelkeeper(model)
model = model.to(device=device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) #optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True, min_lr=1e-4, factor=0.5) #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

for epoch in range(args.epoch):
    #adjust_learning_rate(optimizer, epoch)
    train(epoch)
    test_acc, test_loss = test()
    scheduler.step(test_loss)

    if (1+epoch) % 10 == 0:
        dump_model(epoch, optimizer, model)
