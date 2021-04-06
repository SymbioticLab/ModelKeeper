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
from ImageNet import ImageNet16
from oort.config import oort_config
from oort.matchingopt import Oort

from thirdparty.utils import batchify
from thirdparty.model import AWDRNNModel
from thirdparty.train import train_nlp, eval_nlp
from thirdparty import data
from thirdparty.splitcross import SplitCrossEntropyLoss

logging.basicConfig(level=logging.INFO, filename='./ray_log.e', filemode='w')
logger = logging.getLogger(__name__)


def GenerateConfig(n, path):
    """
    n : number of models
    path : meta file path
    """
    fr = open(path,'rb')
    config_list = pickle.load(fr)

    rng = Random()
    rng.seed(0)
    rng.shuffle(config_list)

    return config_list[:n]
    #return [config_list[i] for i in random.sample(range(0,len(config_list)), n)] 



def train_cv(model, optimizer, criterion, train_loader, device=torch.device("cpu")):
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

    if args.task == "cv":
        if args.data == 'cifar10':
            train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.dataset, train=True, download=True, transform=train_transform),
                batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10(args.dataset, train=False, download=True, transform=test_transform),
                batch_size=args.test_batch_size, shuffle=True, **kwargs)
        elif args.data == 'ImageNet16-120':
            train_data = ImageNet16(args.dataset, True , train_transform, 120)
            test_data  = ImageNet16(args.dataset, False, test_transform, 120)
            assert len(train_data) == 151700 and len(test_data) == 6000
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
            test_loader = torch.utils.data.DataLoader(
                test_data, batch_size=args.test_batch_size, shuffle=True, **kwargs)
        return train_loader, test_loader, None
    elif args.task == "nlp":
        corpus = data.Corpus(args.dataset)
        cuda = 'cuda'
        train_loader = batchify(corpus.train, args.batch_size, args, cuda)
        test_loader = batchify(corpus.test, args.test_batch_size, args, cuda)
        return train_loader, test_loader, len(corpus.dictionary)

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
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
        self.logger = self._create_logger()
        use_cuda = torch.cuda.is_available()

        device = torch.device('cuda')
        if use_cuda:
            for i in range(3, -1, -1):
                try:
                    #device = torch.device('cuda:'+str(i))
                    torch.cuda.set_device(i)
                    self.logger.info(f'End up with cuda device {torch.rand(1).to(device=device)}')
                    break
                except Exception as e:
                    pass #assert(i != 0)

        self.device = torch.device(device if use_cuda else "cpu")


        self.best_acc = 0
        self.best_loss = np.Infinity
        self.train_loader, self.test_loader, self.ntokens = get_data_loaders()
        if args.task == "cv":
            self.model = get_cell_based_tiny_net(conf_list[config['model']])       
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)  # define optimizer
            self.criterion = nn.CrossEntropyLoss()  # define loss function   
        elif args.task == "nlp":
            self.model = AWDRNNModel('CustomRNN', 
                               self.ntokens, 
                               args.emsize, 
                               args.nhid, 
                               args.nlayers, 
                               args.dropout, 
                               args.dropouth, 
                               args.dropouti, 
                               args.dropoute, 
                               args.wdrop, 
                               args.tied,
                               conf_list[config['model']],
                               verbose=False)
            self.criterion = SplitCrossEntropyLoss(args.emsize, splits=[], verbose=False)
            self.params = list(self.model.parameters()) + list(self.criterion.parameters())

            self.optimizer = torch.optim.Adam(self.params, lr=args.lr, weight_decay=args.wdecay)
        self.epoch = 0
        self.model_name = 'model_' + '_'.join([str(val) for val in config.values()]) + '.pth'

        self.use_oort = True
        # Apply Oort to warm start
        if self.use_oort:
            mapper = Oort(oort_config)
            if args.task == "cv":
                weights, num_of_matched = mapper.map_for_model(self.model, torch.rand(8, 3, 32, 32))
            elif args.task == "nlp":
                self.model.eval()
                dummy_input = torch.randint(0, self.ntokens, (70, args.batch_size))
                hidden = self.model.init_hidden(args.batch_size)
                weights, num_of_matched = mapper.map_for_model(self.model, dummy_input, hidden)
            total_layers = 0
            if weights is not None:
                for name, p in self.model.named_parameters():
                    temp_data = (torch.from_numpy(weights[name])).data
                    assert(temp_data.shape == p.data.shape)
                    p.data = temp_data
                    total_layers += 1

            self.logger.info(f"Oort warm starts {num_of_matched} layers, total {total_layers} layers for {self.model_name}")


        self.logger.info(f"Setup for model {self.model_name} ...")
        self.history = {0:{'time':0, 'acc':0, 'loss':0}}

    def _train(self):
        start_time = time.time()
        training_duration, acc, loss = 0, 0, np.Infinity
        if args.task == "cv":
            train_cv(self.model, self.optimizer, self.criterion, self.train_loader, self.device)
            training_duration = time.time() - start_time
            acc, loss = eval_cv(self.model, self.criterion, self.test_loader, self.device)
        elif args.task == "nlp":
            train_nlp(self.model, self.optimizer, self.params, self.criterion, self.train_loader, args, self.epoch,self.device)
            training_duration = time.time() - start_time
            loss = eval_nlp(self.model, self.criterion, self.test_loader, args.test_batch_size, args, self.device)

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
            self.dump_to_zoo(zoo_path = os.path.join(os.environ['HOME'], 'model_zoo', 'nasbench201'))
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
                self.logger.warning(e)

        if METRIC == 'accuracy':
            return {"mean_accuracy": acc}
        else:
            return {"mean_loss": loss}

    def dump_to_zoo(self, zoo_path):
        with open(os.path.join(zoo_path, self.model_name), 'wb') as fout:
            pickle.dump(self.model.to(device='cpu'), fout)
            pickle.dump(self.history, fout)

       if self.use_oort:
            if args.task == "cv":
                torch.onnx.export(self.model, torch.rand(8, 3, 32, 32), os.path.join(zoo_path, f"{self.model_name}.temp_onnx"), 
                                    export_params=True, verbose=0, training=1)
            elif args.task == "nlp":
                self.model.eval()
                with torch.no_grad():
                    dummy_input = torch.randint(0, self.ntokens, (70, args.batch_size))
                    hidden = self.model.init_hidden(args.batch_size)
                    torch.onnx.export(self.model, (dummy_input, hidden), os.path.join(zoo_path, f"{self.model_name}.temp_onnx"), 
                                        export_params=True, verbose=0, training=0,
                                        input_names=['dummy_input'],
                                        output_names=['output'],
                                        dynamic_axes={'dummy_input': [0], 'output': [0]}       
                                    )
            # avoid conflicts
            os.system(f'mv {os.path.join(zoo_path, f"{self.model_name}.temp_onnx")} {os.path.join(zoo_path, f"{self.model_name}.onnx")}')

    def _save(self, checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, "model.pth")
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
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--num_models', type=int, default=2, metavar='N',
                        help='number of models to train (default: 2)')
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
    parser.add_argument('--meta', type=str, default='/gpfs/gpfs0/groups/chowdhury/dywsjtu/')
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")
    parser.add_argument(
        "--address",
        default="localhost:6379",
        help="Address of Ray cluster for seamless distributed execution.")
    ## nlp branch args
    parser.add_argument('--task', type=str, default='cv')
    parser.add_argument('--emsize', type=int, default=400,
                    help='emsize')
    parser.add_argument('--nhid', type=int, default=600,
                        help='nhid')
    parser.add_argument('--nlayers', type=int, default=3,
                        help='nlayers')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='dropout')
    parser.add_argument('--dropouth', type=float, default=0.25,
                        help='dropouth')
    parser.add_argument('--dropouti', type=float, default=0.4,
                        help='dropouti')
    parser.add_argument('--dropoute', type=float, default=0.1,
                        help='dropoute')
    parser.add_argument('--wdrop', type=float, default=0.5,
                        help='wdrop')
    parser.add_argument('--alpha', type=float, default=2,
                        help='alpha')
    parser.add_argument('--beta', type=float, default=1,
                        help='beta')
    parser.add_argument('--bptt', type=float, default=70,
                        help='bptt')
    parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay')
    parser.add_argument('--tied', action='store_true', default=True,
                    help='tied')
    parser.add_argument('--clip', type=float, default=0.25,
                    help='clip')
    args = parser.parse_args()


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
    GRACE_PERIOD = 2#4
    CPU_RESOURCES_PER_TRIAL = 10
    GPU_RESOURCES_PER_TRIAL = 1
    METRIC = 'loss'  # or 'loss'

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
            stop=TrialPlateauStopper(metric='mean_loss', mode='min', std=5e-3,
                            num_results=GRACE_PERIOD+2, grace_period=GRACE_PERIOD),#{"training_epoch": 1 if args.smoke_test else TRAINING_EPOCH},
            resources_per_trial={
                "cpu": CPU_RESOURCES_PER_TRIAL,
                "gpu": GPU_RESOURCES_PER_TRIAL
            },
            #num_samples=args.num_models,
            verbose=3,
            checkpoint_at_end=True,
            checkpoint_freq=1,
            max_failures=3,
            config=CONFIG,
            #local_dir=log_dir,
        )

    if METRIC=='accuracy':
        print("Best config is:", analysis.get_best_config(metric="mean_accuracy"))
    else:
        print("Best config is:", analysis.get_best_config(metric="mean_loss"))
