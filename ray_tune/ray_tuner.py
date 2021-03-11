import os
import time
import argparse
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import ray
from ray import tune
from ray.tune import track, run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler

import torchvision.models as models
from torch.autograd import Variable
from torchvision import datasets, transforms


from nas_201_api import NASBench201API as API
from models import get_cell_based_tiny_net




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


###################################
##  set main configurations here ##

TRAINING_ITERATION = 64
NUM_SAMPLES = 100
REDUCTION_FACTOR = 4
GRACE_PERIOD = 4
CPU_RESOURCES_PER_TRIAL = 1
GPU_RESOURCES_PER_TRIAL = 0
METRIC = 'accuracy'  # or 'loss'

CONFIG = {
    "model": tune.sample_from(lambda _: np.random.randint(133, 135)),
    }
###################################



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

def get_data_loaders():
    train_transform = transforms.Compose(
                [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])


    kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}

    if args.data == 'cifar10':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset, train=True, download=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset, train=False, download=True, transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader

class TrainModel(tune.Trainable):
    """
    Ray Tune's class-based API for hyperparameter tuning
    Note: See https://ray.readthedocs.io/en/latest/_modules/ray/tune/trainable.html#Trainable
    
    """
    def _setup(self, config):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.train_loader, self.test_loader = get_data_loaders()
        self.model = get_cell_based_tiny_net(api.get_net_config(config['model'], args.data))  
        self.model_name = 'model_' + '_'.join([str(val) for val in config.values()]) + '.pth'
        self.best_acc = 0
        self.best_loss = np.Infinity
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)  # define optimizer
        self.criterion = nn.CrossEntropyLoss()  # define loss function
        self.epoch = 0

    def _train(self):
        train(self.model, self.optimizer, self.criterion, self.train_loader, self.device)
        acc, loss = eval(self.model, self.criterion, self.test_loader, self.device)
        self.epoch += 1

        # remember best metric and save checkpoint
        if METRIC == 'accuracy':
            is_best = acc > self.best_acc
        else:
            is_best = loss < self.best_loss
        self.best_acc = max(acc, self.best_acc)
        self.best_loss = min(loss, self.best_loss)
        if is_best:
            try:
                torch.save({
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, self.model_name)
            except Exception as e:
                logger.warning(e)

        if METRIC == 'accuracy':
            return {"mean_accuracy": acc}
        else:
            return {"mean_loss": loss}

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path

    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


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
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--ray-address",
        default="localhost:6379",
        help="Address of Ray cluster for seamless distributed execution.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    api = API(args.meta, verbose=False)
    ray.init(address=args.ray_address)

    if METRIC=='accuracy':
        sched = AsyncHyperBandScheduler(time_attr="training_iteration", 
                                        metric="mean_accuracy", 
                                        mode='max', 
                                        reduction_factor=REDUCTION_FACTOR, 
                                        grace_period=GRACE_PERIOD)
    else:
        sched = AsyncHyperBandScheduler(time_attr="training_iteration", 
                                        metric="mean_loss", 
                                        mode='min', 
                                        reduction_factor=REDUCTION_FACTOR, 
                                        grace_period=GRACE_PERIOD)

    analysis = tune.run(
        TrainModel,
        scheduler=sched,
        queue_trials=True,
        stop={"training_iteration": 1 if args.smoke_test else TRAINING_ITERATION
        },
        resources_per_trial={
            "cpu": CPU_RESOURCES_PER_TRIAL,
            "gpu": GPU_RESOURCES_PER_TRIAL
        },
        num_samples=2,
        verbose=1,
        checkpoint_at_end=True,
        checkpoint_freq=1,
        max_failures=3,
        config=CONFIG
        )

    if METRIC=='accuracy':
        print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
    else:
        print("Best config is:", analysis.get_best_config(metric="mean_loss"))

    # if args.cuda and torch.cuda.is_available():
    #     print("Use cuda")
    #     train(model, optimizer, criterion, train_loader, torch.device("cuda"))
    #     eval(model, criterion, test_loader, torch.device("cuda"))

    # else:
    #     print("Use cpu")
    #     train(model, optimizer, criterion, train_loader, torch.device("cpu"))
    #     eval(model, criterion, test_loader, torch.device("cpu"))