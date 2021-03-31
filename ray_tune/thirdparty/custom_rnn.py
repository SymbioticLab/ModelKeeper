import torch
import torch.nn
import networkx as nx

from thirdparty.multilinear import MultiLinear
import math

class CustomRNNCell(torch.nn.Module):
    
    elementwise_ops_dict = {
        'prod': torch.mul,
        'sum': torch.add
    }
    
    def __init__(self, input_size, hidden_size, recepie):
        super(CustomRNNCell, self).__init__()
        
        self.activations_dict = {
            'tanh': torch.nn.Tanh(),
            'sigm': torch.nn.Sigmoid(),
            'leaky_relu': torch.nn.LeakyReLU()
        }
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recepie = recepie
        self.hidden_tuple_size = 0
        
        components_dict = {}    
    
        self.G = nx.DiGraph()
        for k in recepie.keys():
            if k not in components_dict:
                
                component = self._make_component(recepie[k])
                if component is not None:
                    components_dict[k] = component 
                if k.startswith('h_new'):
                    suffix = k.replace('h_new_', '')
                    if suffix.isdigit():
                        self.hidden_tuple_size = max([self.hidden_tuple_size, int(suffix) + 1])
                
                if k not in self.G.nodes():
                    self.G.add_node(k)
                for i, n in enumerate(recepie[k]['input']):
                    if n not in self.G.nodes():
                        self.G.add_node(k)
                    self.G.add_edge(n, k)

        self.components = torch.nn.ModuleDict(components_dict)
        self.nodes_order = list(nx.algorithms.dag.topological_sort(self.G))
        
    def forward(self, x, hidden_tuple):
        calculated_nodes = {}
        for n in self.nodes_order:
            if n == 'x':
                calculated_nodes['x'] = x.unsqueeze(0)
            elif n.startswith('h_prev') and n.replace('h_prev_', '').isdigit():
                calculated_nodes[n] = hidden_tuple[int(n.replace('h_prev_', ''))].unsqueeze(0)
            elif n in self.components:
                inputs = [calculated_nodes[k] for k in self.recepie[n]['input']]
                calculated_nodes[n] = self.components[n](*inputs)
            else:
                # simple operations
                op = self.recepie[n]['op']
                inputs = [calculated_nodes[k] for k in self.recepie[n]['input']]
                if op in ['elementwise_prod', 'elementwise_sum']:
                    op_func = CustomRNNCell.elementwise_ops_dict[op.replace('elementwise_', '')]
                    calculated_nodes[n] = op_func(inputs[0], inputs[1])
                    for inp in range(2, len(inputs)):
                        calculated_nodes[n] = op_func(calculated_nodes[n], inputs[i])
                elif op == 'blend':
                    calculated_nodes[n] = inputs[0]*inputs[1] + (1 - inputs[0])*inputs[2]
                elif op.startswith('activation'):
                    op_func = self.activations_dict[op.replace('activation_', '')]
                    calculated_nodes[n] = op_func(inputs[0])
        return tuple([calculated_nodes[f'h_new_{i}'][0] for i in range(self.hidden_tuple_size)])
    
    def _make_component(self, spec):
        if spec['op'] == 'linear':
            input_sizes = [self.input_size if inp=='x' else self.hidden_size for inp in spec['input']]
            return MultiLinear(input_sizes, self.hidden_size)


class CustomRNN(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, recepie):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell = CustomRNNCell(input_size, hidden_size, recepie)
        self.reset_parameters()
        
    def forward(self, inputs, hidden_tuple=None):
        batch_size = inputs.size(1)
        if hidden_tuple is None:
            hidden_tuple = tuple([self.init_hidden(batch_size) for _ in range(self.cell.hidden_tuple_size)])
        
        self.check_hidden_size(hidden_tuple, batch_size)
        
        hidden_tuple = tuple([x[0] for x in hidden_tuple])
        outputs = []
        for x in torch.unbind(inputs, dim=0):
            hidden_tuple = self.cell(x, hidden_tuple)
            outputs.append(hidden_tuple[0].clone())

        return torch.stack(outputs, dim=0), tuple([x.unsqueeze(0) for x in hidden_tuple])
    
    def init_hidden(self, batch_size):
        # num_layers == const (1)
        return torch.zeros(1, batch_size, self.hidden_size).to(next(self.parameters()).device)
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            torch.nn.init.uniform_(param, -stdv, stdv)
            
    def check_hidden_size(self, hidden_tuple, batch_size):
        expected_hidden_size = (1, batch_size, self.hidden_size)
        msg = 'Expected hidden size {}, got {}'
        for hx in hidden_tuple:
            if hx.size() != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

