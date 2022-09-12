
import logging

#from tensorboardX import SummaryWriter
import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from onnx import numpy_helper

# from model_arc import make_dot
# from graphviz import Source


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
        # try:
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
        # except RuntimeError:
        #     logging.info(x.size())

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
        self.fc1 = nn.Linear(32 * 3 * 3, self.num_of_class)
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

    def define_wider(self):
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48 * 3 * 3, self.num_of_class)

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
        self.fc1 = nn.Linear(48 * 3 * 3, self.num_of_class)
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


def print_graph(g, level=0):
    if g is None:
        return
    print('*' * level * 4, g)
    for subg in g.next_functions:
        print_graph(subg[0], level + 1)


num_of_class = 10
if __name__ == "__main__":
    dummy_input = torch.rand(16, 3, 32, 32)
    model = Net(num_of_class)
    y = model(dummy_input)
    print(y)
    # print_graph(y.mean().grad_fn, 0)
    #torch.onnx.export(model, dummy_input, "/gpfs/gpfs0/groups/chowdhury/fanlai/net_transformer/Net2Net/model.onnx", export_params=True)

    onnx_model = onnx.load('./model.onnx')

    graph = onnx_model.graph
    initalizers = dict()
    for init in graph.initializer:
        initalizers[init.name] = numpy_helper.to_array(init)

    for name, p in model.named_parameters():
        p.data = (torch.from_numpy(initalizers[name])).data

    print(model(dummy_input))
    #make_dot(y, params=dict(model.named_parameters())).render("model", format="pdf")

    # model_types = ['resnet50']#, 'densenet161', 'shufflenet_v2_x1_0',
    # 'mobilenet_v2']

    # for m in model_types:
    #     model = models.__dict__[m](num_classes=10)
    #     y = model(dummy_input)
    #     print(make_dot(y.mean(), params=dict(model.named_parameters())))
    # with SummaryWriter(comment=m) as w:
    #     w.add_graph(model, (dummy_input, ))
