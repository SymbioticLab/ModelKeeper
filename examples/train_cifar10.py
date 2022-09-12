import argparse
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

sys.path.append('../')
import copy
import logging
import pickle

import numpy as np
import torchvision.models as models
from torchsummary import summary
from utils import NLL_loss_instance, PlotLearning

from net2net import deeper, wider

log_path = '/gpfs/gpfs0/groups/chowdhury/fanlai/net_transformer/Net2Net/examples/logging'
with open(log_path, 'w') as fout:
    pass

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(log_path, mode='a'),
                    logging.StreamHandler()
                ])

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging status')
parser.add_argument('--noise', type=float, default=5e-2,
                    help='noise or no noise 0-1')
parser.add_argument('--data', type=str, default='cifar10')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
learning_rate = args.lr

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
        datasets.CIFAR10('/gpfs/gpfs0/groups/chowdhury/fanlai/dataset', train=True, download=True, transform=train_transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('/gpfs/gpfs0/groups/chowdhury/fanlai/dataset', train=False, download=True, transform=test_transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

elif args.data == 'cifar100':
    num_of_class = 100
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/gpfs/gpfs0/groups/chowdhury/fanlai/dataset', train=True, download=True, transform=train_transform),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('/gpfs/gpfs0/groups/chowdhury/fanlai/dataset', train=False, download=True, transform=test_transform),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

class Net(nn.Module):
    def __init__(self, num_of_class=10):
        super(Net, self).__init__()
        self.num_of_class = num_of_class

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.AvgPool2d(5, 1)
        self.fc1 = nn.Linear(32 * 3 * 3, self.num_of_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.fill_(0.0)
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0.0)

    def forward(self, x):
        try:
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
            x = self.fc1(x)
            return F.log_softmax(x)
        except RuntimeError:
            logging.info(x.size())

    def net2net_wider(self):
        self.conv1, self.conv2, self.bn1 = wider(self.conv1, self.conv2, 12,
                                          self.bn1, noise_var=args.noise)
        self.conv2, self.conv3, self.bn2 = wider(self.conv2, self.conv3, 24,
                                          self.bn2, noise_var=args.noise)
        self.conv3, self.fc1, self.bn3 = wider(self.conv3, self.fc1, 48,
                                        self.bn3, noise_var=args.noise)
        logging.info(self)

    def net2net_deeper(self):
        s = deeper(self.conv1, nn.ReLU, bnorm_flag=True, noise_var=args.noise)
        self.conv1 = s
        s = deeper(self.conv2, nn.ReLU, bnorm_flag=True, noise_var=args.noise)
        self.conv2 = s
        s = deeper(self.conv3, nn.ReLU, bnorm_flag=True, noise_var=args.noise)
        self.conv3 = s
        logging.info(self)

    def define_deeper(self):
        self.conv1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(),
                                   nn.Conv2d(8, 8, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, kernel_size=3, padding=1))
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*3*3, self.num_of_class)
        logging.info(self)

    def manual_deeper(self):
        self.conv1 = nn.Sequential(self.conv1,
                                   nn.BatchNorm2d(8),
                                   nn.ReLU(),
                                   nn.Conv2d(8, 8, kernel_size=3, padding=1))
        self.conv2 = nn.Sequential(self.conv2,
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(),
                                   nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.conv3 = nn.Sequential(self.conv3,
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(),
                                   nn.Conv2d(32, 32, kernel_size=3, padding=1))
        logging.info(self)

    def net2net_deeper_nononline(self):
        s = deeper(self.conv1, None, bnorm_flag=True, noise_var=args.noise)
        self.conv1 = s
        s = deeper(self.conv2, None, bnorm_flag=True, noise_var=args.noise)
        self.conv2 = s
        s = deeper(self.conv3, None, bnorm_flag=True, noise_var=args.noise)
        self.conv3 = s
        logging.info(self)

    def define_wider(self):
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48*3*3, self.num_of_class)

    def define_wider_deeper(self):
        self.conv1 = nn.Sequential(nn.Conv2d(3, 12, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(12),
                                   nn.ReLU(),
                                   nn.Conv2d(12, 12, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Sequential(nn.Conv2d(12, 24, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(24),
                                   nn.ReLU(),
                                   nn.Conv2d(24, 24, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU(),
                                   nn.Conv2d(48, 48, kernel_size=3, padding=1))
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48*3*3, self.num_of_class)
        logging.info(self)


def net2net_deeper_recursive(model):
    """
    Apply deeper operator recursively any conv layer.
    """
    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            s = deeper(module, nn.ReLU, bnorm_flag=False)
            model._modules[name] = s
        elif isinstance(module, nn.Sequential):
            module = net2net_deeper_recursive(module)
            model._modules[name] = module
    return model

def adjust_learning_rate(optimizer, decay=0.95):
    global learning_rate
    for param_group in optimizer.param_groups:
        learning_rate *= decay
        param_group['lr'] = learning_rate#param_group['lr'] * decay

def train(epoch):
    global optimizer
    #adjust_learning_rate(optimizer)
    model.train()

    avg_loss = 0
    avg_accu = 0.

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        avg_accu += pred.eq(target.data.view_as(pred)).cpu().sum()
        avg_loss += loss.item()
        if (batch_idx+1) % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

        # we give a quick test of initial model
        if epoch == 0 and batch_idx == 10: break 

    avg_loss /= (batch_idx + 1)
    avg_accu /= len(train_loader.dataset)

    logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item(), avg_accu * 100.))

    return avg_accu, avg_loss

def test():
    model.eval()
    test_loss = 0
    correct = 0.
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    logging.info('Test set: Average loss: {}, Accuracy: {}/{} ({}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * float(correct) / float(len(test_loader.dataset))))
    return correct / len(test_loader.dataset), test_loss


def run_training(model, run_name, epochs, plot=None):
    global optimizer

    model.cuda()
    accu_test, loss_test = test()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    if plot is None:
        plot = PlotLearning('./plots/cifar/', 10, prefix=run_name)
    for epoch in range(epochs + 1):
        accu_train, loss_train = train(epoch)
        accu_test, loss_test = test()
        logs = {}
        logs['acc'] = accu_train
        logs['val_acc'] = accu_test
        logs['loss'] = loss_train
        logs['val_loss'] = loss_test
        plot.plot(logs)
    return plot

if __name__ == "__main__":
    start_t = time.time()
    criterion = nn.NLLLoss()
    logging.info("\n\n > Teacher training ... ")
    model = Net(num_of_class)
    plot = run_training(model, 'Teacher_', args.epochs+1)

    with open('mother_cifar10_fixlr.pkl', 'wb') as fout:
        pickle.dump(model.to(device='cpu'), fout)

    # with open('mother_cifar10.pkl', 'rb') as fin:
    #     model = pickle.load(fin)

    model_ = copy.deepcopy(model)

    # wider student training
    logging.info("\n\n > Wider Student training ... ")

    model = copy.deepcopy(model_)
    model.net2net_wider()
    plot = run_training(model, 'Wider_student_', args.epochs + 1)

    logging.info("\n\n > Deeper Student training ... ")

    model = copy.deepcopy(model_)
    model.net2net_deeper()
    plot = run_training(model, 'Deeper_student_', args.epochs + 1)


    logging.info("\n\n > Wider teacher training ... ")
    model = copy.deepcopy(model_)
    model.define_wider()
    plot = run_training(model, 'Deeper_teacher_', args.epochs + 1)

    logging.info("\n\n > Deeper teacher training ... ")
    model = copy.deepcopy(model_)   
    model.define_deeper()
    plot = run_training(model, 'Deeper_teacher_', args.epochs + 1)

    logging.info("\n\n > Deeper manual teacher training ... ")
    model = copy.deepcopy(model_)   
    model.manual_deeper()
    plot = run_training(model, 'Deeper_teacher_', args.epochs + 1)
