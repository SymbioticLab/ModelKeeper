import os
import time
import random
import pickle
import argparse
import logging
import numpy as np
import collections
import pandas
import string

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import ray
from ray import tune
from ray.tune import track, run_experiments
from ray.tune.schedulers import AsyncHyperBandScheduler, FIFOScheduler
from onlinescheduler import OnlineScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper
#import torchvision.models as models
from torch.autograd import Variable
from torchvision import datasets, transforms

import socket
from random import Random
from collections import defaultdict, deque

import sys
from ImageNet import ImageNet16

# ModelKeeper dependency
from modelkeeper.config import modelkeeper_config
from modelkeeper.clientservice import ModelKeeperClient
from modelkeeper.matchingopt import ModelKeeper

# sys.path.insert(0, '../ray_tune/')
# Imgclsmob zoo
from models.torchcv.model_provider import get_model as ptcv_get_model
# Cifar zoo
from models.cifarmodels.model_provider import get_cv_model
from models.nasbench import get_cell_based_tiny_net
import threading

ray.tune.ray_trial_executor.DEFAULT_GET_TIMEOUT = 600

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

    modelidx_base = 0

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
        output = model(data)
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
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            loss = criterion(output, target)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            avg_loss += loss.item()

    accuracy = float(correct) / total
    model.to(device='cpu')

    return accuracy, avg_loss/len(data_loader)

def get_data_loaders(train_bz, test_bz):

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

    if args.data == 'cifar10':

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset, train=True, download=True, transform=train_transform),
            batch_size=train_bz, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dataset, train=False, download=True, transform=test_transform),
            batch_size=test_bz, shuffle=True, **kwargs)
    elif args.data == 'cifar100':
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.dataset, train=True, download=True, transform=train_transform),
            batch_size=train_bz, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.dataset, train=False, download=True, transform=test_transform),
            batch_size=test_bz, shuffle=True, **kwargs)

    elif args.data == 'ImageNet16-120':
        train_data = ImageNet16(os.path.join(args.dataset, 'ImageNet16-120'), True , train_transform, 32, 120)
        test_data  = ImageNet16(os.path.join(args.dataset, 'ImageNet16-120'), False, test_transform, 32, 120)
        #assert len(train_data) == 151700 and len(test_data) == 6000
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=train_bz, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=test_bz, shuffle=True, **kwargs)

    return train_loader, test_loader, None


def polish_name(model_name):
    updated_name = ''
    for c in model_name:
        if c in string.punctuation:
            updated_name += '_'
        else:
            updated_name += c

    return updated_name.replace(' ', '_').replace('__', '_')


def get_str_type(input_str):
    input_str = input_str.strip()

    if input_str.isdigit():
        return int(input_str)
    if input_str == 'True' or input_str == 'False':
        return eval(input_str)
    if '.' in input_str:
        return float(input_str)
    return str(input_str)


def get_model(temp_model_name):
    model_type = temp_model_name.split('(')[0].strip()
    temp_args = [x.strip() for x in temp_model_name.split('(')[1].replace(')','').strip().split(',') if len(x.strip())>0]
    args_model = {}
    for pair in temp_args:
        [_key, _value] = pair.split('=')

        args_model[_key.strip()] = get_str_type(_value)

    args_model['name'] = model_type
    return args_model

def change_opt_lr(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr


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


class BestAccuracyStopper(Stopper):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, verbose=False, delta=0, trace_func=logging.info):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = collections.defaultdict(int)
        self.best_score = {}
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, trial_id: str, result: dict):

        score = result['mean_accuracy']

        if trial_id not in self.best_score:
            self.best_score[trial_id] = score
            #self.save_checkpoint(val_loss, model)
        elif score <= self.best_score[trial_id] + self.delta:
            self.counter[trial_id] += 1
            self.trace_func(f'EarlyStopping counter: {self.counter[trial_id]} out of {self.patience}')
            if self.counter[trial_id] >= self.patience:
                return True
        else:
            self.best_score[trial_id] = score
            self.counter[trial_id] = 0

        return False

    def stop_all(self):
        return False


class TrainModel(tune.Trainable):
    """
    Ray Tune's class-based API for hyperparameter tuning
    Note: See https://ray.readthedocs.io/en/latest/_modules/ray/tune/trainable.html#Trainable

    """

    def setup(self, config):
        self.logger = self.creat_my_log()
        use_cuda = torch.cuda.is_available()

        device = torch.device('cuda')
        seed = 1

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = True

        self.device = device if use_cuda else torch.device("cpu")
        self.train_loader, self.test_loader, self.ntokens = get_data_loaders(args.batch_size, args.test_batch_size)

        temp_model_name = config['config']['name']
        if args.task == "nasbench":
            self.model = get_cell_based_tiny_net(conf_list[temp_model_name])
        elif args.task == 'torchcv':

            num_labels = {'cifar10': 10, "cifar100": 100, "ImageNet16-120": 120}
            num_classes = num_labels[args.data]

            if '(' in temp_model_name:
                args_model = get_model(temp_model_name)
                args_model['num_classes'] = num_classes

                self.model = get_cv_model(**args_model)
            else:
                self.model = ptcv_get_model(temp_model_name, pretrained=False, num_classes=num_classes)
        else:
            assert("Have not implemented!")


        self.model_name = polish_name(temp_model_name) #'model_' + '_'.join([str(val) for val in config.values()])
        self.export_path = self.model_name + '.onnx'

        arrival_time = config['config'].get('arrival', 0)
        self.logger.info(f"Setup for model {self.model_name} ..., supposed {arrival_time}")

        self.use_keeper = args.use_keeper

        # Apply ModelKeeper to warm start
        if self.use_keeper:
            start_matching = time.time()

            # Create a session to modelkeeper server
            modelkeeper_client = ModelKeeperClient(modelkeeper_config)

            self.model.eval()
            self.model.to(device='cpu')

            dummy_input = torch.rand(2, 3, 32, 32)
            # export to onnx format; Some models may be slow in onnx
            torch.onnx.export(self.model, dummy_input, self.export_path,
                export_params=True, verbose=0, training=1, do_constant_folding=False)

            weights, self.meta_info = modelkeeper_client.query_for_model(self.export_path)

            if weights is not None:
                for name, p in self.model.named_parameters():
                    try:
                        temp_data = (torch.from_numpy(weights[name])).data
                        assert(temp_data.shape == p.data.shape)
                        p.data = temp_data.to(dtype=p.data.dtype)
                    except Exception as e:
                        self.logger.error(f"Fail to load weight for {self.model_name}, as {e}")

            self.logger.info(f"ModelKeeper warm starts {self.model_name} in {int(time.time() - start_matching)} sec, meta: {self.meta_info}")
            modelkeeper_client.stop()


        self.best_acc = 0
        self.best_loss = np.Infinity
        self.epoch = 0

        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay, momentum=args.momentum, nesterov=True)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', patience=4,
                        verbose=True, min_lr=1e-3, factor=0.5, threshold=0.01)

        self.history = {0:{'time':0, 'acc':0, 'loss':0}}


    def step(self):
        start_time = time.time()

        if self.use_keeper and self.meta_info:
            # the first two epoch to warm up
            if self.epoch == 0:
                warm_up_lr = args.lr * max(0.5, 1.-self.meta_info['num_of_matched']/self.meta_info['parent_layers'])
                change_opt_lr(self.optimizer, warm_up_lr)
            # roll back to the original lr
            elif self.epoch == 2:
                change_opt_lr(self.optimizer, args.lr)

        # Automatically change batch size if OOM
        recovery_trials = 3
        for i in range(1, 1+recovery_trials):
            try:
                train_cv(self.model, self.optimizer, self.criterion, self.train_loader, self.device, self.scheduler)
                break
            except Exception as e:
                train_bz = test_bz = max(4, args.batch_size//(i*2))
                self.logger.info(f"Model {self.model_name} fails {e}, change batch size to {train_bz}")
                self.train_loader, self.test_loader, self.ntokens = get_data_loaders(train_bz, train_bz)

        training_duration = time.time() - start_time
        acc, loss = eval_cv(self.model, self.criterion, self.test_loader, self.device)
        self.scheduler.step(acc)

        self.epoch += 1
        self.history[self.epoch] = {
                    'time': self.history[self.epoch-1]['time']+training_duration,
                    'acc': acc,
                    'loss': loss,
                    'epoch': self.epoch
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


    def stop(self):
        self.model.to(device='cpu')
        self.model.eval()

        local_path = f"{os.environ['HOME']}/experiment/ray_zoos"
        os.makedirs(local_path, exist_ok=True)
        export_path = os.path.join(local_path, self.export_path)
        dummy_input = torch.rand((2, 3, 32, 32))

        with open(export_path, 'wb') as fout:
            pickle.dump(self.model, fout)
            pickle.dump(dummy_input, fout)

        if args.use_keeper:
            # Call the offline API to register the model
            os.system(f"nohup python {os.environ['HOME']}/experiment/ModelKeeper/ray_tune/keeper_offline.py --model_file={export_path} --accuracy={self.history[self.epoch]['acc']} &")
            self.logger.info("Call keeper offline register API")

        self.logger.info(f"Training of {self.model_name} completed with {self.history[self.epoch]}")


    def creat_my_log(self):
        log_dir = f"{os.environ['HOME']}/experiment/ray_logs"
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        host_name = str(socket.gethostname()).split('.')[0]
        log_path = os.path.join(log_dir, host_name)

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
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_models', type=int, default=300, metavar='N',
                        help='number of models to train ')
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging status')
    parser.add_argument('--data', type=str, default='cifar100')
    parser.add_argument('--dataset', type=str, default='/users/fanlai/experiment/data')
    parser.add_argument('--trace', type=str, default='/users/fanlai/experiment/ModelKeeper/ray_tune/workloads/torchcv_list.csv')
    parser.add_argument('--meta', type=str, default='/users/fanlai/experiment/data')
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--address",
        default="10.0.0.1:6379",
        help="Address of Ray cluster for seamless distributed execution.")

    ## nlp branch args
    parser.add_argument('--task', type=str, default='nasbench')
    parser.add_argument('--use_keeper', type=bool, default=False)

    args, unknown = parser.parse_known_args()
    keeper_service = None

    logging.info(args)

    if args.use_keeper:
        logging.info(modelkeeper_config)
        keeper_service = ModelKeeper(modelkeeper_config)
        keeper_service.start_service()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    conf_list = GenerateConfig(args.num_models, os.path.join(args.meta, args.data + "_config.pkl"))

    # Clear the log dir
    log_dir = f"{os.environ['HOME']}/experiment/ray_logs"
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    os.system(f"rm {os.environ['HOME']}/experiment/ray_logs/*")

    ###################################
    ##  set main configurations here ##

    TRAINING_EPOCH = 1#32

    REDUCTION_FACTOR = 1.000001
    GRACE_PERIOD = 7
    CPU_RESOURCES_PER_TRIAL = 1
    GPU_RESOURCES_PER_TRIAL = 1
    METRIC = 'accuracy'  # or 'loss'

    if args.task == "torchcv":
        temp_conf = []

        workload = pandas.read_csv(args.trace)
        for row in workload.sort_values(by="arrival").itertuples():
            temp_conf.append({'name': row.name,'arrival': row.arrival})
    else:
        temp_conf = [{'name':'model_'+str(n) for n in range(args.num_models)}] #list(range(args.num_models))

    CONFIG = {
        "config": tune.grid_search(temp_conf),
    }
    ray.init(address=f"{args.address}")

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
            scheduler=OnlineScheduler(FIFOScheduler()),
            queue_trials=True,
            #stop={"training_epoch": 1},
            stop=CombinedStopper(
                MaximumIterationStopper(max_iter=args.epochs),
                BestAccuracyStopper(),
                TrialPlateauStopper(metric='mean_accuracy', mode='max', std=2e-3,
                num_results=10, grace_period=GRACE_PERIOD),
            ),
            resources_per_trial={
                "cpu": CPU_RESOURCES_PER_TRIAL,
                "gpu": GPU_RESOURCES_PER_TRIAL
            },
            verbose=3,
            checkpoint_at_end=False,
            checkpoint_freq=10000,
            max_failures=3,
            config=CONFIG,
            local_dir=os.path.join(os.environ['HOME'], 'experiment/ray_results')
        )

    if METRIC=='accuracy':
        logging.info("Best config is:", analysis.get_best_config(metric="mean_accuracy", mode='max'))
    else:
        logging.info("Best config is:", analysis.get_best_config(metric="mean_loss", mode='min'))

    if keeper_service is not None:
        keeper_service.stop_service()

