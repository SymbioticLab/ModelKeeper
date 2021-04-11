import torch
import torch.nn
import torch.nn.functional as F

import math

class MultiLinear(torch.nn.Module):

    def __init__(self, input_sizes, output_size):
        super(MultiLinear, self).__init__()
        self.input_sizes = input_sizes
        self.output_size = output_size
        
        weights = []
        for input_size in input_sizes:
            weights.append(torch.nn.Parameter(torch.Tensor(output_size, input_size)))
        self.weights = torch.nn.ParameterList(weights)
        
        self.bias = torch.nn.Parameter(torch.Tensor(output_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.weights)):
            torch.nn.init.kaiming_uniform_(self.weights[i], a=math.sqrt(5))
        
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights[0])
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, *inputs):
        result = F.linear(inputs[0], self.weights[0], self.bias)
        for i in range(1, len(self.weights)):
            result = result + F.linear(inputs[i], self.weights[i])            
        return result

    def extra_repr(self):
        return 'input_sizes={}, output_size={}'.format(
            self.input_sizes, self.output_size
        )
