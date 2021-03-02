import onnx
from onnx import numpy_helper
import numpy

def load_model_data(file):
    onnx_model = onnx.load(file+".onnx")
    graph = onnx_model.graph

    layer_weights = dict()
    for init in graph.initializer:
        layer_weights[init.name] = numpy_helper.to_array(init)

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

        self.parent_weights = load_model_data(parent['name'])
        self.child_weights = load_model_data(child['name'])

    def get_child_layers(self, graph, node_id):
        """Get the trainable next layers"""
        visited = set()
        ret = []

        def dfs(node):
            visited.add(node)
            # Trainable tensor. TODO: solve BN
            if len(graph.nodes[node]['attr']['dims']) > 1:
                ret.append(node)
            else:
                [dfs(edge[1]) for edge in graph.out_edges(node) if edge[1] not in visited]
        dfs(node_id)
        return ret

    def cascading_mapping(self):
        """
           We need to assure the following layers have been assigned,
           such that widening proceeding layers can replicate the right units
        """ 
        
        mappings = self.mapping_indices.copy()
        mappings.reverse()

        for (parent_layer, child_layer) in mappings:
            # get child layers
            following_layers = self.get_child_layers(child_layer)
            # operation on layers, override child weights
            pass

    def e2e_mapping(self):
        """
            Handle unmapped layers by padding identity layers or random initialization
        """
        pass


