import onnx
from onnx import numpy_helper
import numpy
from nettransformer import widen, widen_child
import logging
import os

def load_model_data(file):
    onnx_model = onnx.load(file+".onnx")
    graph = onnx_model.graph

    layer_weights = dict()
    for init in graph.initializer:
        layer_weights[init.name] = numpy.copy(numpy_helper.to_array(init))

    return layer_weights

class MappingOperator(object):
    """Map parent weights to child weights given the mapping index"""

    def __init__(self, parent, child, mapping_indices, zoo_path):
        """
        @ parent, child: graph of nodes for each model
        @ mapping_indices: (parent_layer_name, child_layer_name): sorted by the topological order of child
        """
        self.parent = parent
        self.child = child
        self.mapping_indices = mapping_indices

        self.parent_weights = load_model_data(os.path.join(zoo_path, parent.graph['name']))
        self.child_weights = load_model_data(os.path.join(zoo_path, child.graph['name']))

    def get_child_layers(self, graph, node_id):
        """Get the trainable next layers"""
        visited = set()
        ret = []

        def dfs(node):
            visited.add(node)
            # Trainable tensor. TODO: solve BN
            if len(graph.nodes[node]['attr']['dims']) > 1 and node != node_id:
                ret.append(graph.nodes[node]['attr']['layer_name'])
            else:
                [dfs(edge[1]) for edge in graph.out_edges(node) if edge[1] not in visited]

        dfs(node_id)
        return ret

    def get_weights(self, graph, initializer, node):
        layer_dims = graph.nodes[node]['attr']['dims']
        if len(layer_dims) < 2:
            return None, None

        layer_name = graph.nodes[node]['attr']['layer_name']
        weight = initializer[layer_name+'.weight']
        bias = initializer[layer_name+'.bias']

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
                print('Skip mapping {} ({}) to {} ({})'.format(parent_layer_name, self.parent.nodes[parent_layer]['attr']['op_type'], 
                                                            child_layer_name, self.child.nodes[child_layer]['attr']['op_type']))
            else:
                n_weight, n_bias, mapping_index = widen(parent_w, parent_b, child_w, child_b, noise_factor=5e-2)

                assert(n_weight.shape == child_w.shape and n_bias.shape == child_b.shape)

                self.child_weights[child_layer_name+'.weight'] = n_weight
                self.child_weights[child_layer_name+'.bias'] = n_biass

                # get its child layers
                following_layers = self.get_child_layers(self.child, child_layer)
                for layer in following_layers:
                    layer_w = self.child_weights[layer+'.weight']
                    nl_weight = widen_child(layer_w, mapping_index, noise_factor=5e-2)

                    assert(layer_w.shape == nl_weight.shape)
                    self.child_weights[layer+'.weight'] = nl_weight

                # operation on layers, override child weights
                print('Successfully map {} ({}) to {} ({})'.format(parent_layer_name, self.parent.nodes[parent_layer]['attr']['dims'], 
                                                            child_layer_name, self.child.nodes[child_layer]['attr']['dims']))

    def get_mapping_weights(self):
        return self.child_weights

    def e2e_mapping(self):
        """
            Handle unmapped layers by padding identity layers or random initialization
        """
        
        pass


