import numpy as np


class RecepieGenerator:

    def __init__(
        self, 
        hidden_tuple_size=2,
        intermediate_vertices=7,
        main_operations = ['linear', 'blend', 'elementwise_prod', 'elementwise_sum'],
        main_weights = [3., 1., 1., 1.],
        activations = ['activation_tanh', 'activation_sigm', 'activation_leaky_relu'],
        activation_weights = [1., 1., 1.],
        linear_connections = [2, 3],
        linear_connections_weights = [4, 1]
        ):
        self.hidden_tuple_size = hidden_tuple_size
        self.intermediate_vertices = intermediate_vertices
        self.main_operations = main_operations
        self.main_probabilities = np.array(main_weights)/np.sum(main_weights)
        self.activations = activations
        self.activation_probabilities = np.array(activation_weights)/np.sum(activation_weights)
        self.linear_connections = linear_connections
        self.linear_connections_probabilities = np.array(linear_connections_weights)/np.sum(linear_connections_weights)

    def _generate_redundant_graph(self, recepie, base_nodes):
        i = 0
        activation_nodes = []
        while i < self.hidden_tuple_size + self.intermediate_vertices:
            op = np.random.choice(self.main_operations, 1, p=self.main_probabilities)[0]
            if op == 'linear':
                num_connections = np.random.choice(self.linear_connections, 1, 
                                                   p=self.linear_connections_probabilities)[0]
                connection_candidates = base_nodes + activation_nodes
                if num_connections > len(connection_candidates):
                    num_connections = len(connection_candidates)
                
                connections = np.random.choice(connection_candidates, num_connections, replace=False)
                recepie[f'node_{i}'] = {'op':op, 'input':connections}
                i += 1
                
                # after linear force add activation node tied to the new node, if possible (nodes budget)
                op = np.random.choice(self.activations, 1, p=self.activation_probabilities)[0]
                recepie[f'node_{i}'] = {'op':op, 'input':[f'node_{i - 1}']}
                activation_nodes.append(f'node_{i}')
                i += 1
                
            elif op in ['blend', 'elementwise_prod', 'elementwise_sum']:
                # inputs must exclude x
                if op == 'blend':
                    num_connections = 3
                else:
                    num_connections = 2
                connection_candidates = list(set(base_nodes) - set('x')) + list(recepie.keys())
                if num_connections <= len(connection_candidates):
                    connections = np.random.choice(connection_candidates, num_connections, replace=False)
                    recepie[f'node_{i}'] = {'op':op, 'input':connections}
                    i += 1

    def _create_hidden_nodes(self, recepie):
        new_hiddens_map = {}
        for k in np.random.choice(list(recepie.keys()), self.hidden_tuple_size, replace=False):
            new_hiddens_map[k] = f'h_new_{len(new_hiddens_map)}'
            
        for k in new_hiddens_map:
            recepie[new_hiddens_map[k]] = recepie[k]
            del recepie[k]
            
        for k in recepie:
            recepie[k]['input'] = [new_hiddens_map.get(x, x) for x in recepie[k]['input']]

    def _remove_redundant_nodes(self, recepie):
        q = [f'h_new_{i}' for i in range(self.hidden_tuple_size)]
        visited = set(q)
        while len(q) > 0:
            if q[0] in recepie:
                for node in recepie[q[0]]['input']:
                    if node not in visited:
                        q.append(node)
                        visited.add(node)
            q = q[1:]

        for k in list(recepie.keys()):
            if k not in visited:
                del recepie[k]

        return visited

    def generate_random_recepie(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        prev_hidden_nodes = [f'h_prev_{i}' for i in range(self.hidden_tuple_size)]
        base_nodes = ['x'] + prev_hidden_nodes
        
        recepie = {}
        self._generate_redundant_graph(recepie, base_nodes)
        self._create_hidden_nodes(recepie)
        visited = self._remove_redundant_nodes(recepie)

        is_sanity_check_ok = True

        # check that all input nodes are in the graph
        for node in base_nodes:
            if node not in visited:
                is_sanity_check_ok = False
                break

        # constraint: prev hidden nodes are not connected directly to new hidden nodes
        for i in range(self.hidden_tuple_size):
            if len(set(recepie[f'h_new_{i}']['input']) & set(prev_hidden_nodes)) > 0:
                is_sanity_check_ok = False
                break

        return recepie, is_sanity_check_ok

    def get_example_recepie(self, name):
        if name == 'rnn':
            recepie = {
                'f':{'op':'linear', 'input':['x', 'h_prev_0']},
                'h_new_0':{'op':'activation_tanh', 'input':['f']}
            }
        elif name == 'lstm':
            recepie = {
                'i':{'op':'linear', 'input':['x', 'h_prev_0']},
                'i_act':{'op':'activation_tanh', 'input':['i']},
                
                'j':{'op':'linear', 'input':['x', 'h_prev_0']},
                'j_act':{'op':'activation_sigm', 'input':['j']},
                
                'f':{'op':'linear', 'input':['x', 'h_prev_0']},
                'f_act':{'op':'activation_sigm', 'input':['f']},
                
                'o':{'op':'linear', 'input':['x', 'h_prev_0']},
                'o_act':{'op':'activation_tanh', 'input':['o']},
                
                'h_new_1_part1':{'op':'elementwise_prod', 'input':['f_act', 'h_prev_1']},
                'h_new_1_part2':{'op':'elementwise_prod', 'input':['i_act', 'j_act']},
                
                'h_new_1':{'op':'elementwise_sum', 'input':['h_new_1_part1', 'h_new_1_part2']},
                
                'h_new_1_act':{'op':'activation_tanh', 'input':['h_new_1']},
                'h_new_0':{'op':'elementwise_prod', 'input':['h_new_1_act', 'o_act']}
            }
        elif name == 'gru':
            recepie = {
                'r':{'op':'linear', 'input':['x', 'h_prev_0']},
                'r_act':{'op':'activation_sigm', 'input':['r']},
                
                'z':{'op':'linear', 'input':['x', 'h_prev_0']},
                'z_act':{'op':'activation_sigm', 'input':['z']},
                
                'rh':{'op':'elementwise_prod', 'input':['r_act', 'h_prev_0']},
                'h_tilde':{'op':'linear', 'input':['x', 'rh']},
                'h_tilde_act':{'op':'activation_tanh', 'input':['h_tilde']},
                
                'h_new_0':{'op':'blend', 'input':['z_act', 'h_prev_0', 'h_tilde_act']}
            }
        else:
            raise Exception(f'Unknown recepie name: {name}')
        return recepie

