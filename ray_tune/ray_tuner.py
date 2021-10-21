import os
import time
import random
import pickle
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

import socket
from random import Random
from collections import defaultdict, deque

import sys
from vgg import *
from ImageNet import ImageNet16

# ModelKeeper dependency
from modelkeeper.config import modelkeeper_config
from modelkeeper.clientservice import ModelKeeperClient

from thirdparty.utils import batchify
from thirdparty.model import AWDRNNModel
from thirdparty.train import train_nlp, eval_nlp
from thirdparty import data
from thirdparty.splitcross import SplitCrossEntropyLoss

logging.basicConfig(level=logging.INFO, filename='./ray_log.e', filemode='w')
logger = logging.getLogger(__name__)

modelidx_base = 1150

def GenerateConfig(n, path):
    """
    n : number of models
    path : meta file path
    """
    if args.task == "v100":
        config_list = vgg_zoo()
    else:
        fr = open(path,'rb')
        config_list = pickle.load(fr)

    rng = Random()
    rng.seed(0)
    rng.shuffle(config_list)

    return config_list[modelidx_base:modelidx_base+n]
    #return [config_list[i] for i in random.sample(range(0,len(config_list)), n)] 



def train_cv(model, optimizer, criterion, train_loader, device=torch.device("cpu"), scheduler=None):
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

    model.to(device='cpu')

def eval_cv(model, criterion, data_loader, device=torch.device("cpu")):
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
    correct = avg_loss = 0.
    total = 0   
    with torch.no_grad():
        for data, target in data_loader:
            data, target = Variable(data.to(device)), Variable(target.to(device))
            _, output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, target)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            avg_loss += loss.item()

    accuracy = correct / total
    model.to(device='cpu')

    return accuracy, avg_loss/len(data_loader)

def get_data_loaders():

    if 'ImageNet' in args.data:
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std  = [x / 255 for x in [63.22,  61.26 , 65.09]]
        lists = [transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=2), 
                transforms.ToTensor(), 
                transforms.Normalize(mean, std)]
        train_transform = transforms.Compose(lists)
        test_transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(), 
                transforms.RandomCrop(32, padding=2), 
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        test_transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])



    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}

    if args.task == "nasbench":
        if args.data == 'cifar10':

            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.dataset, train=True, download=True, transform=train_transform),
                batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.dataset, train=False, download=True, transform=test_transform),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)

        elif args.data == 'ImageNet16-120':
            train_data = ImageNet16(os.path.join(args.dataset, 'ImageNet16-120'), True , train_transform, 32, 120)
            test_data  = ImageNet16(os.path.join(args.dataset, 'ImageNet16-120'), False, test_transform, 32, 120)
            #assert len(train_data) == 151700 and len(test_data) == 6000
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        return train_loader, test_loader, None


from ray.tune.stopper import Stopper
class TrialPlateauStopper(Stopper):
    """Early stop single trials when they reached a plateau.

    When the standard deviation of the `metric` result of a trial is
    below a threshold `std`, the trial plateaued and will be stopped
    early.

    Args:
        metric (str): Metric to check for convergence.
        std (float): Maximum metric standard deviation to decide if a
            trial plateaued. Defaults to 0.01.
        num_results (int): Number of results to consider for stdev
            calculation.
        grace_period (int): Minimum number of timesteps before a trial
            can be early stopped
        metric_threshold (Optional[float]):
            Minimum or maximum value the result has to exceed before it can
            be stopped early.
        mode (Optional[str]): If a `metric_threshold` argument has been
            passed, this must be one of [min, max]. Specifies if we optimize
            for a large metric (max) or a small metric (min). If max, the
            `metric_threshold` has to be exceeded, if min the value has to
            be lower than `metric_threshold` in order to early stop.
    """

    def __init__(self,
                 metric: str,
                 std: float = 0.01,
                 num_results: int = 4,
                 grace_period: int = 4,
                 metric_threshold: float = None,
                 mode: str = 'max'):
        self._metric = metric
        self._mode = mode

        self._std = std
        self._num_results = num_results
        self._grace_period = grace_period
        self._metric_threshold = metric_threshold

        if self._metric_threshold:
            if mode not in ["min", "max"]:
                raise ValueError(
                    f"When specifying a `metric_threshold`, the `mode` "
                    f"argument has to be one of [min, max]. "
                    f"Got: {mode}")

        self._iter = defaultdict(lambda: 0)
        self._trial_results = defaultdict(
            lambda: deque(maxlen=self._num_results))

    def __call__(self, trial_id: str, result: dict):
        metric_result = result.get(self._metric)
        self._trial_results[trial_id].append(metric_result)
        self._iter[trial_id] += 1

        # If still in grace period, do not stop yet
        if self._iter[trial_id] < self._grace_period:
            return False

        # If not enough results yet, do not stop yet
        if len(self._trial_results[trial_id]) < self._num_results:
            return False

        # If metric threshold value not reached, do not stop yet
        if self._metric_threshold is not None:
            if self._mode == "min" and metric_result > self._metric_threshold:
                return False
            elif self._mode == "max" and \
                    metric_result < self._metric_threshold:
                return False

        # Calculate stdev of last `num_results` results
        try:
            current_std = np.std(self._trial_results[trial_id])
        except Exception:
            current_std = float("inf")

        # If stdev is lower than threshold, stop early.
        return current_std < self._std

    def stop_all(self):
        return False


class TrainModel(tune.Trainable):
    """
    Ray Tune's class-based API for hyperparameter tuning
    Note: See https://ray.readthedocs.io/en/latest/_modules/ray/tune/trainable.html#Trainable
    
    """

    def _setup(self, config):
        self.logger = self._create_logger()
        use_cuda = torch.cuda.is_available()

        device = torch.device('cuda')

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        
        self.device = device if use_cuda else torch.device("cpu")
        self.train_loader, self.test_loader, self.ntokens = get_data_loaders()

        if args.task == "nasbench":
            self.model = get_cell_based_tiny_net(conf_list[config['model']])
        else:
            assert("Have not implemented!")



        self.model_name = 'model_' + '_'.join([str(val) for val in config.values()])
        self.export_path = self.model_name + '.onnx'

        self.use_oort = True # True
        learning_rate = args.lr

        # Apply ModelKeeper to warm start
        if self.use_oort:
            start_matching = time.time()

            # Create a session to modelkeeper server
            modelkeeper_client = ModelKeeperClient(modelkeeper_config)

            self.model.eval()
            self.model.to(device='cpu')

            dummy_input = torch.rand(2, 3, 32, 32)
            # export to onnx format
            torch.onnx.export(self.model, dummy_input, self.export_path,
                export_params=True, verbose=0, training=1, do_constant_folding=False)

            weights, meta_info = modelkeeper_client.query_for_model(self.export_path)

            for name, p in self.model.named_parameters():
                if weights is not None:
                    temp_data = (torch.from_numpy(weights[name])).data
                    assert(temp_data.shape == p.data.shape)
                    p.data = temp_data.to(dtype=p.data.dtype)

            self.logger.info(f"ModelKeeper warm starts {self.model_name} in {int(time.time() - start_matching)} sec, meta: {meta_info}")
            modelkeeper_client.stop()

            learning_rate *= (1.-meta_info.get('matching_score'))

        self.best_acc = 0
        self.best_loss = np.Infinity
        self.epoch = 0

        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True, min_lr=5e-4, factor=0.5)  
        
        self.logger.info(f"Setup for model {self.model_name} ...")
        self.history = {0:{'time':0, 'acc':0, 'loss':0}}


    def _train(self):
        start_time = time.time()

        if args.task == "nasbench":
            train_cv(self.model, self.optimizer, self.criterion, self.train_loader, self.device, self.scheduler)
        
        training_duration = time.time() - start_time
        acc, loss = eval_cv(self.model, self.criterion, self.test_loader, self.device)
        self.scheduler.step(loss)

        self.epoch += 1
        self.history[self.epoch] = {
                    'time': self.history[self.epoch-1]['time']+training_duration,
                    'acc': acc,
                    'loss': loss,
                }

        self.logger.info(f"Trained model {self.model_name}: epoch {self.epoch}, acc {acc}, loss {loss}")

        # remember best metric and save checkpoint
        if METRIC == 'accuracy':
            is_best = acc > self.best_acc
        else:
            is_best = loss < self.best_loss
        self.best_acc = max(acc, self.best_acc)
        self.best_loss = min(loss, self.best_loss)

        if METRIC == 'accuracy':
            return {"mean_accuracy": acc}
        else:
            return {"mean_loss": loss}


    def _stop(self):

        zoo_path = os.path.join(os.environ['HOME'], 'model_zoo', 'nasbench201')
        os.makedirs(zoo_path, exist_ok=True)


        self.model.to(device='cpu')
        self.model.eval()


        if self.use_oort:
            if args.task == "nasbench":
                torch.onnx.export(self.model, torch.rand(2, 3, 32, 32), self.export_path, export_params=True, verbose=0, training=1)

            # register model to the zoo
            modelkeeper_client = ModelKeeperClient(modelkeeper_config)
            modelkeeper_client.register_model_to_zoo(self.export_path, accuracy=self.history[self.epoch]['acc'])
            modelkeeper_client.stop()
            os.remove(self.export_path)

        self.logger.info(f"Training of {self.model_name} completed ...")


    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
        torch.save(self.model.state_dict(), checkpoint_path)
        return checkpoint_path


    def _restore(self, checkpoint_path):
        self.model.load_state_dict(torch.load(checkpoint_path))


    def _create_logger(self):
        log_dir = f"{os.environ['HOME']}/ray_logs"
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'{socket.gethostname()}')

        logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(log_path, mode='a'),
                        logging.StreamHandler()
                    ])
        return logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Cifar10 Example")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_models', type=int, default=300, metavar='N',
                        help='number of models to train ')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
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
    parser.add_argument('--data', type=str, default='ImageNet16-120')
    parser.add_argument('--dataset', type=str, default='/users/fanlai/data')
    parser.add_argument('--meta', type=str, default='/users/fanlai/data')
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--address",
        default="localhost:6379",
        help="Address of Ray cluster for seamless distributed execution.")

    ## nlp branch args
    parser.add_argument('--task', type=str, default='nasbench')
    

    args, unknown = parser.parse_known_args()


    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    conf_list = GenerateConfig(args.num_models, args.meta + args.data + "_config.pkl")

    # Clear the log dir
    log_dir = f"{os.environ['HOME']}/ray_logs"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    os.system(f"rm {os.environ['HOME']}/ray_logs/*")

    ###################################
    ##  set main configurations here ##

    TRAINING_EPOCH = 1#32

    REDUCTION_FACTOR = 1.000001
    GRACE_PERIOD = 5#4
    CPU_RESOURCES_PER_TRIAL = 10
    GPU_RESOURCES_PER_TRIAL = 1
    METRIC = 'accuracy'  # or 'loss'

    CONFIG = {
        "model": tune.grid_search(list(range(args.num_models))),
    }
    ray.init(address="auto")

    if METRIC=='accuracy':
        sched = AsyncHyperBandScheduler(time_attr="training_epoch", 
                                        metric="mean_accuracy", 
                                        mode='max', 
                                        reduction_factor=REDUCTION_FACTOR, 
                                        grace_period=GRACE_PERIOD,
                                        brackets=1)
    else:
        sched = AsyncHyperBandScheduler(time_attr="training_epoch", 
                                        metric="mean_loss", 
                                        mode='min', 
                                        reduction_factor=REDUCTION_FACTOR, 
                                        grace_period=GRACE_PERIOD,
                                        brackets=1)

    # Random scheduler (FIFO) that trains all models to the end
    analysis = tune.run(
            TrainModel,
            #scheduler=sched,
            queue_trials=True,
            #stop={"training_epoch": 1},
            stop=TrialPlateauStopper(metric='mean_accuracy', mode='max', std=4e-3,
                num_results=GRACE_PERIOD+2, grace_period=GRACE_PERIOD),#{"training_epoch": 1 if args.smoke_test else TRAINING_EPOCH},
            resources_per_trial={
                "cpu": CPU_RESOURCES_PER_TRIAL,
                "gpu": GPU_RESOURCES_PER_TRIAL
            },
            #num_samples=args.num_models,
            verbose=3,
            checkpoint_at_end=False,
            checkpoint_freq=10000,
            max_failures=3,
            config=CONFIG,
        )

    if METRIC=='accuracy':
        print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
    else:
        print("Best config is:", analysis.get_best_config(metric="mean_loss"))
