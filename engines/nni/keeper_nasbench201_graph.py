import click
import nni
import nni.retiarii.evaluator.pytorch.lightning as PL
# import torch.nn as nn
import torchmetrics
import torch
import os
from nni.retiarii import model_wrapper, serialize, serialize_cls
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig
# from nni.retiarii.nn.pytorch import NasBench201Cell
from nni.retiarii.nn.pytorch.utils import generate_new_label, get_fixed_value
from nni.retiarii.strategy import Random, RegularizedEvolution
from pytorch_lightning.callbacks import LearningRateMonitor
from timm.optim import RMSpropTF
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torchvision.datasets import CIFAR100
from collections import OrderedDict
from typing import Callable, List, Union, Tuple, Optional
from threading import Thread
# from base_ops import ResNetBasicblock, PRIMITIVES, OPS_WITH_STRIDE
import random
import pytorch_lightning
from collections import defaultdict, deque
from nni.experiment import RemoteMachineConfig
import pickle as pkl
import numpy as np
import time
import logging
from nni.retiarii import Mutator
from nni.retiarii import basic_unit
import nni.retiarii.nn.pytorch as nn
# ModelKeeper dependency
from modelkeeper.config import modelkeeper_config
from modelkeeper.clientservice import ModelKeeperClient
from modelkeeper.matcher import ModelKeeper

import json
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# import torch
# import torch.nn as nn


OPS_WITH_STRIDE = {
    'none': lambda C_in, C_out, stride: Zero(C_in, C_out, stride),
    'avg_pool_3x3': lambda C_in, C_out, stride: Pooling(C_in, C_out, stride, 'avg'),
    # 'max_pool_3x3': lambda C_in, C_out, stride: Pooling(C_in, C_out, stride, 'max'),
    'conv_3x3': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (3, 3), (stride, stride), (1, 1), (1, 1)),
    'conv_1x1': lambda C_in, C_out, stride: ReLUConvBN(C_in, C_out, (1, 1), (stride, stride), (0, 0), (1, 1)),
    'skip_connect': lambda C_in, C_out, stride: nn.Identity() if stride == 1 and C_in == C_out
    else FactorizedReduce(C_in, C_out, stride),
}

PRIMITIVES = ['none', 'avg_pool_3x3', 'conv_3x3', 'conv_1x1', 'skip_connect']
@basic_unit
class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out)
        )

    def forward(self, x):
        return self.op(x)
@basic_unit
class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out),
        )

    def forward(self, x):
        return self.op(x)
@basic_unit
class Pooling(nn.Module):
    def __init__(self, C_in, C_out, stride, mode):
        super().__init__()
        if C_in == C_out:
            self.preprocess = None
        else:
            self.preprocess = ReLUConvBN(C_in, C_out, 1, 1, 0, 1)
        if mode == 'avg':
            self.op = nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False)
        elif mode == 'max':
            self.op = nn.MaxPool2d(3, stride=stride, padding=1)
        else:
            raise ValueError('Invalid mode={:} in Pooling'.format(mode))

    def forward(self, x):
        if self.preprocess is not None:
            x = self.preprocess(x)
        return self.op(x)
@basic_unit
class Zero(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.is_zero = True

    def forward(self, x):
        if self.C_in == self.C_out:
            if self.stride == 1:
                return x.mul(0.)
            else:
                return x[:, :, ::self.stride, ::self.stride].mul(0.)
        else:
            shape = list(x.shape)
            shape[1] = self.C_out
            zeros = x.new_zeros(shape, dtype=x.dtype, device=x.device)
            return zeros
@basic_unit
class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out
        self.relu = nn.ReLU(inplace=False)
        if stride == 2:
            C_outs = [C_out // 2, C_out - C_out // 2]
            self.convs = nn.ModuleList()
            for i in range(2):
                self.convs.append(nn.Conv2d(C_in, C_outs[i], 1, stride=stride, padding=0, bias=False))
            self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)
        else:
            raise ValueError('Invalid stride : {:}'.format(stride))
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.convs[0](x), self.convs[1](y[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out
@basic_unit
class ResNetBasicblock(nn.Module):
    def __init__(self, inplanes, planes, stride):
        super().__init__()
        assert stride == 1 or stride == 2, 'invalid stride {:}'.format(stride)
        self.conv_a = ReLUConvBN(inplanes, planes, 3, stride, 1, 1)
        self.conv_b = ReLUConvBN(planes, planes, 3, 1, 1, 1)
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False))
        elif inplanes != planes:
            self.downsample = ReLUConvBN(inplanes, planes, 1, 1, 0, 1)
        else:
            self.downsample = None
        self.in_dim = inplanes
        self.out_dim = planes
        self.stride = stride
        self.num_conv = 2

    def forward(self, inputs):
        basicblock = self.conv_a(inputs)
        basicblock = self.conv_b(basicblock)

        if self.downsample is not None:
            inputs = self.downsample(inputs)  # residual
        return inputs + basicblock


class CellMutator(Mutator):
    def __init__(self, targets: 'List[str]'):
        super().__init__()
        self.targets = targets

    # def retrieve_targeted_graphs(self, graph: 'Graph') -> 'List[Graph]':
    #     return [graph.find_node(t) for t in self.targets]
        # return model.get_nodes_by_label(self.target)

    def mutate(self, graph):
        # target_nodes = self.retrieve_targeted_graphs(graph)
        target_nodes = graph.get_nodes_by_label(self.targets)

        operations = {f'{i}_{j}': self.choice(list(OPS_WITH_STRIDE.keys())) for i in range(4) for j in range(i + 1, 4)}
        for target_node in target_nodes:
            target_node.update_operation(None, op_dict=operations)

class NasBench201Cell(nn.Module):
    """
    Cell structure that is proposed in NAS-Bench-201 [nasbench201]_ .
    This cell is a densely connected DAG with ``num_tensors`` nodes, where each node is tensor.
    For every i < j, there is an edge from i-th node to j-th node.
    Each edge in this DAG is associated with an operation transforming the hidden state from the source node
    to the target node. All possible operations are selected from a predefined operation set, defined in ``op_candidates``.
    Each of the ``op_candidates`` should be a callable that accepts input dimension and output dimension,
    and returns a ``Module``.
    Input of this cell should be of shape :math:`[N, C_{in}, *]`, while output should be :math:`[N, C_{out}, *]`. For example,
    The space size of this cell would be :math:`|op|^{N(N-1)/2}`, where :math:`|op|` is the number of operation candidates,
    and :math:`N` is defined by ``num_tensors``.
    Parameters
    ----------
    op_candidates : list of callable
        Operation candidates. Each should be a function accepts input feature and output feature, returning nn.Module.
    in_features : int
        Input dimension of cell.
    out_features : int
        Output dimension of cell.
    num_tensors : int
        Number of tensors in the cell (input included). Default: 4
    label : str
        Identifier of the cell. Cell sharing the same label will semantically share the same choice.
    References
    ----------
    .. [nasbench201] Dong, X. and Yang, Y., 2020. Nas-bench-201: Extending the scope of reproducible neural architecture search.
        arXiv preprint arXiv:2001.00326.
    """

    @staticmethod
    def _make_dict(x):
        if isinstance(x, list):
            return OrderedDict([(str(i), t) for i, t in enumerate(x)])
        return OrderedDict(x)

    def __init__(self, op_candidates,
                 in_features: int, out_features: int, num_tensors: int = 4, stride: int = 1,
                 label: Optional[str] = None):
        super().__init__()
        self._label = generate_new_label(label)
        self.stride = stride
        self.layers = nn.ModuleList()
        self.in_features = in_features
        self.out_features = out_features
        self.num_tensors = num_tensors
        op_candidates = self._make_dict(op_candidates)


        for tid in range(1, num_tensors):
            node_ops = nn.ModuleList()
            for j in range(tid):
                inp = in_features if j == 0 else out_features
                op_choices = OrderedDict([(key, cls(inp, out_features, stride))
                                          for key, cls in op_candidates.items()])
                node_ops.append(nni.retiarii.nn.pytorch.LayerChoice(op_choices, label=f'{self._label}__{j}_{tid}'))  # put __ here to be compatible with base engine
            self.layers.append(node_ops)

    def forward(self, inputs):
        tensors = [inputs]
        for layer in self.layers:
            current_tensor = []
            for i, op in enumerate(layer):
                current_tensor.append(op(tensors[i]))
            current_tensor = torch.sum(torch.stack(current_tensor), 0)
            tensors.append(current_tensor)
        return tensors[-1]

# @model_wrapper
class NasBench201(nn.Module):
    def __init__(self,
                 stem_out_channels: int = 16,
                 num_modules_per_stack: int = 5,
                 num_labels: int = 100):
        super().__init__()
        self.channels = C = stem_out_channels
        self.num_modules = N = num_modules_per_stack
        self.num_labels = num_labels

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        layer_channels = [C] * N + [C * 2] + [C * 2] * N + [C * 4] + [C * 4] * N
        layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

        C_prev = C
        self.cells = nn.ModuleList()
        for C_curr, reduction in zip(layer_channels, layer_reductions):
            if reduction:
                cell = ResNetBasicblock(C_prev, C_curr, 2)
            else:
                # cell = NasBench201Cell({prim: lambda C_in, C_out: OPS_WITH_STRIDE[prim](C_in, C_out, 1) for prim in PRIMITIVES},
                #                        C_prev, C_curr, label='cell')
                cell = NasBench201Cell(OPS_WITH_STRIDE, C_prev, C_curr, label='cell')
            self.cells.append(cell)
            C_prev = C_curr

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(C_prev),
            nn.ReLU(inplace=True)
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, self.num_labels)

    def forward(self, inputs):
        feature = self.stem(inputs)
        for cell in self.cells:
            feature = cell(feature)

        out = self.lastact(feature)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)

        return logits


class AccuracyWithLogits(torchmetrics.Accuracy):
    def update(self, pred, target):
        return super().update(torch.nn.functional.softmax(pred), target)


@serialize_cls
class NasBench201TrainingModule(PL.LightningModule):
    def __init__(self, max_epochs=200, learning_rate=5e-3, weight_decay=5e-4, grace_period=4, std=0.01, 
        num_results=10, use_keeper=False, register_runtime=1e10):
        super().__init__()
        self.save_hyperparameters('learning_rate', 'weight_decay', 'max_epochs')
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = AccuracyWithLogits()
        self.std = std
        self.num_results = num_results
        self.grace_period = grace_period
        self.round_result = deque(maxlen=self.num_results)
        self.round = 0
        self.flag = False
        self.use_keeper = use_keeper
        self.start_time = None
        self.register_runtime = register_runtime

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        self.log('val_loss', self.criterion(y_hat, y), prog_bar=True)
        self.log('val_accuracy', self.accuracy(y_hat, y), prog_bar=True)

    def configure_optimizers(self):
        # optimizer = RMSpropTF(self.parameters(), lr=self.hparams.learning_rate,
        #                       weight_decay=self.hparams.weight_decay,
        #                       momentum=0.9, alpha=0.9, eps=1.0)
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=0.9)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4, verbose=True, min_lr=1e-6, factor=0.5, threshold=0.02)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            # val_checkpoint_on is val_loss passed in as checkpoint_on
            'monitor': 'val_accuracy'
        }

        if self.use_keeper:
            start_matching = time.time()
            # self.model_name = os.getcwd().split('/')[-2]

            print("====", os.getcwd(), os.getcwd().split('/'))
            conf_file = '/' + '/'.join(os.getcwd().split('/')[:-1])+'/parameter.cfg'
            script = json.load(open(conf_file))["parameters"]["model_script"]

            #self.model_name = os.getcwd().split('/')[-2]
            choices = ["layerchoice_cell__0_1_", "layerchoice_cell__0_2_", "layerchoice_cell__1_2_", "layerchoice_cell__0_3_", "layerchoice_cell__1_3_", "layerchoice_cell__2_3_"]
            config = []
            for choice in choices:
                result = script.find(choice)
                config.append(script[result: result+len(choice)+18].split()[0].split("_", 5)[-1])
            self.model_name = '-'.join(config)
            print(f"** {self.model_name}**")
            logging.info(f"Try to warm start {self.model_name}")
            # Create a session to modelkeeper server
            modelkeeper_client = ModelKeeperClient(modelkeeper_config)

            zoo_path = os.path.join(os.environ['HOME'], 'experiment', 'nas-zoo')
            os.makedirs(zoo_path, exist_ok=True)
            self.export_path = os.path.join(zoo_path, self.model_name+".onnx")

            dummy_input = torch.rand(2, 3, 32, 32)
            # export to onnx format; Some models may be slow in onnx
            torch.onnx.export(self.model.cpu(), dummy_input, self.export_path,
                export_params=True, verbose=0, training=1, do_constant_folding=False)
            weights, self.meta_info = modelkeeper_client.query_for_model(self.export_path)

            if weights is not None:
                for name, p in self.model.named_parameters():
                    try:
                        temp_data = (torch.from_numpy(weights[name])).data
                        assert(temp_data.shape == p.data.shape)
                        p.data = temp_data.to(dtype=p.data.dtype)
                    except Exception as e:
                        logging.info(f"Fail to load weight for {self.model_name}, as {e}")

            logging.info(f"ModelKeeper warm starts {self.model_name} in {int(time.time() - start_matching)} sec, meta: {self.meta_info}")
            self.model.cuda()
            modelkeeper_client.stop()
            self.start_time = time.time()

        return [optimizer], [scheduler]
        # return {
        #     'optimizer': optimizer,
        #     # 'scheduler': CosineAnnealingLR(optimizer, self.hparams.max_epochs)#todo
        #     # 'scheduler':  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4, verbose=True, min_lr=1e-6, factor=0.5, threshold=0.02)
        # }

    def on_validation_epoch_end(self):
        self.round += 1
        shutdown = False
        val_acc = self.trainer.callback_metrics['val_accuracy'].item()
        self.round_result.append(val_acc)
        if self.round > self.grace_period:
            try:
                current_std = np.std(self.round_result)
            except Exception:
                current_std = float("inf")
            if current_std < self.std or self.round > 2:
                shutdown = True
            print(val_acc)
        nni.report_intermediate_result(val_acc)
        if shutdown:
            self.teardown("shutdown")
        self.register_model()

    def register_model(self):
        if self.use_keeper and self.flag is False and (time.time()-self.start_time)/60. > self.register_runtime:
            dummy_input = torch.rand(8, 3, 32, 32)
            # model_name = os.getcwd().split('/')[-2]
            export_path = os.path.join(os.environ['HOME'], 'experiment/nas-zoo', self.model_name+".onnx")
            torch.onnx.export(self.model.cpu(), dummy_input, export_path, export_params=True, verbose=0, training=1)
            self.model.cuda()
            logging.info(f"Call keeper offline register API {os.path.abspath(os.getcwd())}")
            os.system(f"nohup python {os.path.abspath(os.getcwd())}/keeper_offline.py --model_file={export_path} --accuracy={self.trainer.callback_metrics['val_accuracy'].item()} &")
            logging.info("Call keeper offline register API")
            self.flag = True

    def teardown(self, stage):

        if stage in ['fit', 'shutdown']:
            dummy_input = torch.rand(8, 3, 32, 32)
            # model_name = os.getcwd().split('/')[-2]
            export_path = os.path.join(os.environ['HOME'], 'experiment/nas-zoo', self.model_name+".onnx")
            torch.onnx.export(self.model.cpu(), dummy_input, export_path, export_params=True, verbose=0, training=1)
            self.model.cuda()
            logging.info(f"Call keeper offline register API {os.path.abspath(os.getcwd())}")
            if self.use_keeper and self.flag is False:
                os.system(f"nohup python {os.path.abspath(os.getcwd())}/keeper_offline.py --model_file={export_path} --accuracy={self.trainer.callback_metrics['val_accuracy'].item()} &")
                logging.info("Call keeper offline register API")
                self.flag = True

        # for name, p in self.model.named_parameters():
        #     print(name, p)
        # torch.save(self.model.state_dict(), os.path.join("/users/Yinwei/", model_name+"_weight"))
        if stage == 'fit':
            nni.report_final_result(self.trainer.callback_metrics['val_accuracy'].item())
        if stage == "shutdown":
            nni.report_final_result(self.trainer.callback_metrics['val_accuracy'].item())
            exit()


@click.command()
@click.option('--epochs', default=150, help='Training length.')
@click.option('--batch_size', default=64, help='Batch size.')
@click.option('--port', default=8081, help='On which port the experiment is run.')
@click.option('--benchmark', is_flag=True, default=False)
@click.option('--data', default=f"{os.environ['HOME']}/data", help='Training length.')
@click.option('--use_keeper', is_flag=True, default=False)
@click.option('--register_runtime', default=1e10, help='When to register model to keeper')
def _multi_trial_test(epochs, batch_size, port, benchmark, data, use_keeper, register_runtime):
    # initalize dataset. Note that 50k+10k is used. It's a little different from paper
    transf = [
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]
        # transforms.Normalize([x / 255 for x in [129.3, 124.1, 112.4]], [x / 255 for x in [68.2, 65.4, 70.4]])
    train_dataset = serialize(CIFAR100, data, train=True, download=True, transform=transforms.Compose(transf + normalize))
    test_dataset = serialize(CIFAR100, data, train=False, transform=transforms.Compose(normalize))

    # specify training hyper-parameters
    training_module = NasBench201TrainingModule(max_epochs=epochs, use_keeper=use_keeper, register_runtime=register_runtime)
    # FIXME: need to fix a bug in serializer for this to work
    # lr_monitor = serialize(LearningRateMonitor, logging_interval='step')
    # trainer = PL.Trainer(max_epochs=epochs, gpus=1)
    # early_stop_callback = create_wrapper_cls(EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max"))
    # trainer = PL.Trainer(max_epochs=epochs, gpus=1, callbacks=[early_stop_callback])
    trainer = PL.Trainer(max_epochs=epochs, gpus=1, checkpoint_callback=False, progress_bar_refresh_rate=0)
    lightning = PL.Lightning(
        lightning_module=training_module,
        trainer=trainer,
        train_dataloader=PL.DataLoader(train_dataset, num_workers=4, batch_size=batch_size, shuffle=True),
        val_dataloaders=PL.DataLoader(test_dataset, num_workers=4, batch_size=batch_size),
    )

    #if use_keeper:
    #    keeper_service = ModelKeeper(modelkeeper_config)
    #    keeper_service.start_service()

    # strategy = RegularizedEvolution(optimize_mode='maximize', population_size=15, sample_size=2, cycles=1000, mutation_prob=1e-4, on_failure='ignore')
    strategy = Random()
    # dummy_input = torch.rand(8, 3, 32, 32)
    model = NasBench201()
    pretrained_dict = torch.load("./base_model_weight")
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    exp = RetiariiExperiment(model, lightning, [], strategy)

    # exp_config = RetiariiExeConfig('remote')
    # exp_config.experiment_name = 'nni-nasbench-baseline'
    # exp_config.trial_concurrency = 2
    # exp_config.max_trial_number = 1000
    # exp_config.trial_gpu_number = 1
    # exp_config.training_service.use_active_gpu = True
    # exp_config.training_service.reuse_mode = False
    # exp_config.experiment_working_directory = f"{os.environ['HOME']}/experiment/"

    # confs = []
    # for i in range(1,2):
    #     rm_conf = RemoteMachineConfig()
    #     rm_conf.host = '10.0.0.{}'.format(i)
    #     rm_conf.user = 'Yinwei' ## Change to your username
    #     # rm_conf.password = ''
    #     rm_conf.ssh_key_file = f"{os.environ['HOME']}/.ssh/id_rsa"
    #     rm_conf.python_path = f"{os.environ['HOME']}/experiment/anaconda3/envs/nni-mk/bin"
    #     rm_conf.use_active_gpu = True
    #     rm_conf.max_trial_number_per_gpu = 2
    #     confs.append(rm_conf)
    # exp_config.training_service.machine_list = confs
    # exp_config.execution_engine = 'base'


    exp_config = RetiariiExeConfig('local')
    exp_config.trial_concurrency = 1
    exp_config.max_trial_number = 10
    exp_config.trial_gpu_number = 1
    exp_config.training_service.use_active_gpu = True
    exp_config.training_service.max_trial_number_per_gpu = 2
    exp_config.execution_engine = 'base'
    # if benchmark:
    #     exp_config.benchmark = 'nasbench201-cifar100'
        # exp_config.execution_engine = 'benchmark'
    exp.run(exp_config, port)


if __name__ == '__main__':
    _multi_trial_test()


