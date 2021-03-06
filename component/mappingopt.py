import onnx
from onnx import numpy_helper
import numpy
from nettransformer import widen, widen_child, deepen
import logging
import os
import collections

def load_model_data(file):
    onnx_model = onnx.load(file)
    graph = onnx_model.graph

    layer_weights = dict()
    for init in graph.initializer:
        layer_weights[init.name] = numpy.copy(numpy_helper.to_array(init))

    return layer_weights

class MappingOperator(object):
    """Map parent weights to child weights given the mapping index"""

    def __init__(self, parent, child, mapping_indices):
        """
        @ parent, child: graph of nodes for each model
        @ mapping_indices: (parent_layer_name, child_layer_name): sorted by the topological order of child
        """
        self.parent = parent
        self.child = child
        self.mapping_indices = mapping_indices

        self.parent_weights = load_model_data(parent.graph['name'])
        self.child_weights = load_model_data(child.graph['name'])

        self.reset_layers = set()
        self.num_of_matched = 0

    def get_child_layers(self, graph, node_id):
        """Get the trainable next layers"""
        visited = set()
        ret = []

        def dfs(node):
            visited.add(node)
            # Trainable tensor. TODO: solve BN
            if len(graph.nodes[node]['attr']['dims']) >=1:
                ret.append(graph.nodes[node]['attr']['layer_name'])

            # we overwrite BN and its child layers
            if len(graph.nodes[node]['attr']['dims']) <= 1:
                [dfs(edge[1]) for edge in graph.out_edges(node) if edge[1] not in visited]

        dfs(node_id)
        return ret[1:] # skip node == node_id

    def get_weights(self, graph, initializer, node):
        layer_dims = graph.nodes[node]['attr']['dims']
        if len(layer_dims) < 1:
            return None, None

        layer_name = graph.nodes[node]['attr']['layer_name']
        weight = initializer[layer_name+'.weight']
        bias = initializer[layer_name+'.bias'] if layer_name+'.bias' in initializer else None

        return weight, bias

    def cascading_mapping(self):
        """
           We need to assure the following layers have been assigned,
           such that widening proceeding layers can replicate the right units
        """ 
        
        mappings = self.mapping_indices.copy()
        mappings.reverse()

        for (parent_layer, child_layer) in mappings:
            # Get trainable weights
            parent_w, parent_b = self.get_weights(self.parent, self.parent_weights, parent_layer)
            child_w, child_b = self.get_weights(self.child, self.child_weights, child_layer)

            parent_layer_name = self.parent.nodes[parent_layer]['attr']['layer_name']
            child_layer_name = self.child.nodes[child_layer]['attr']['layer_name']

            if parent_w is None or child_w is None:
                print('Skip mapping {} to {}'.format(self.parent.nodes[parent_layer]['attr']['op_type'], 
                                                    self.child.nodes[child_layer]['attr']['op_type']))
            else:
                n_weight, n_bias, mapping_index = widen(parent_w, parent_b, child_w, child_b, noise_factor=5e-2)

                #assert(n_weight.shape == child_w.shape and n_bias.shape == child_b.shape)

                self.child_weights[child_layer_name+'.weight'] = n_weight
                if n_bias is not None:
                    self.child_weights[child_layer_name+'.bias'] = n_bias

                # get its child layers, and override child weights
                following_layers = self.get_child_layers(self.child, child_layer)
                for layer in following_layers:
                    layer_w = self.child_weights[layer+'.weight']
                    nl_weight = widen_child(layer_w, mapping_index, noise_factor=5e-2)

                    #assert(layer_w.shape == nl_weight.shape)
                    self.child_weights[layer+'.weight'] = nl_weight
                
                self.num_of_matched += 1
                self.reset_layers.add(child_layer_name)
                print('Successfully map {} ({}) to {} ({})'.format(parent_layer_name, self.parent.nodes[parent_layer]['attr']['dims'], 
                                                            child_layer_name, self.child.nodes[child_layer]['attr']['dims']))

    def get_mapping_weights(self):
        return self.child_weights, self.num_of_matched

    def pad_mapping(self, threshold=4):
        """
            Handle unmapped layers by padding identity layers or random initialization
        """
        # reverse the graph, and then run DFS to record the gap between warmed layers and next closest one
        reversed_graph = self.child.reverse(copy=True)
        layer_gaps = collections.defaultdict(int)
        visited = set()
        num_of_padding = 0

        def dfs(graph, node, depth):
            visited.add(node)
            cur_depth = depth
            layer_name, layer_dims = self.child.nodes[node]['attr']['layer_name'], len(self.child.nodes[node]['attr']['dims'])

            if layer_dims > 1:
                cur_depth += 1
                layer_gaps[layer_name] += cur_depth
                if layer_name in self.reset_layers:
                    cur_depth = 0

            for source, target in graph.out_edges(node):
                if target not in visited:
                    dfs(graph, target, cur_depth)

        [dfs(self.child, node, depth=0) for node in self.child.nodes() if self.child.in_degree(node)==0]
        visited = set()
        [dfs(reversed_graph, node, depth=0) for node in reversed_graph.nodes() if reversed_graph.in_degree(node)==0]
        
        for trainable_layer in layer_gaps:
            if layer_gaps[trainable_layer] < threshold and trainable_layer not in self.reset_layers:
                n_weight, n_bias = deepen(self.child_weights[trainable_layer+'.weight'])
                assert(n_weight.shape == self.child_weights[trainable_layer+'.weight'].shape)
                self.child_weights[trainable_layer+'.weight'] = n_weight

                num_of_padding += 1
                print("Pad layer {} with gap {}".format(trainable_layer, layer_gaps[trainable_layer]))

        print("\n\nPad {} identity layers".format(num_of_padding))


