
import onnx
import numpy
import igraph
from igraph import *
import time
import collections

def split_inputs(in_list):
    # input list may contain trainable weights
    input_nodes = []
    i = 0

    while i < len(in_list):
        # tensor nodes are numeric by default
        node_str = in_list[i]
        if node_str.isnumeric(): input_nodes.append(node_str)
        elif node_str.startswith('input.'): pass
        else: break

        i += 1

    return input_nodes, in_list[i:]

def get_tensor_shapes(model_graph):

    node_shapes = dict()

    if model_graph.initializer:
        for init in model_graph.initializer:
            node_shapes[init.name] = tuple(init.dims)
    else:
        for node in model_graph.input:
            node_shapes[node.name] = tuple([p.dim_value for p in node.type.tensor_type.shape.dim])

    return node_shapes

def visualize_graph(graph):
    visual_style = {}
    visual_style["vertex_size"] = 5
    visual_style["layout"] = graph.layout("tree")
    visual_style["bbox"] = (600, 600)
    plot(graph, **visual_style)

def get_layer_view(graph):
    '''
        @source_ids: source nodes in the graph. The default is 0.
        
        @return: BFS order of the graph. Results will guide the search order.
        For nodes in the same layer, we sort by certain order later.
    '''

    in_degrees = dict()
    que = []

    for v in graph.vs:
        deg = graph.indegree(v)
        if deg == 0:
            que.append(v.index)
        else:
            in_degrees[v.index] = deg

    nodes_by_layer = []
    
    while que:
        layer_len = len(que)
        nodes_by_layer.append([])

        for _ in range(layer_len):
            node = que.pop(0)
            nodes_by_layer[-1].append(node)

            for s in graph.successors(node):
                in_degrees[s] -= 1

                if in_degrees[s] == 0:
                    del in_degrees[s]
                    que.append(s)

    assert(len(in_degrees) == 0)

    #print([[graph.vs[v]['attribution']['name'] for v in vs] for vs in nodes_by_layer])
    return nodes_by_layer

class MatchingOperator(object):

    def __init__(self, parent):

        self.parent = parent.copy()

        self.operation_view = dict()
        self.operation_store = collections.defaultdict(list)

        self.init_operation_store()

    def init_operation_store(self):
        '''
            Categorize all operators in parent graph
        '''

        nodes_by_layer = get_layer_view(self.parent)

        for layer_id, vs in enumerate(nodes_by_layer):
            for v in vs:
                self.operation_view[v] = layer_id

        for v in self.parent.vs:
            opt_type = v['attr']['op_type']
            self.operation_store[opt_type].append({'nid':v.index, 'lid':self.operation_view[v.index], 
                                                'dim':v['attr']['dim']})

        for opt in self.operation_store:
            self.operation_store[opt].sort(key=lambda k:k['lid'])

    def get_candidate_store(self, child):
        '''
            @return: possible mapping of each node in child
        '''
        candidates = dict()
        
        for v in child.vs:
            op_type = v['attr']['op_type']
            candidates[v.index] = self.operation_store[op_type].copy()
            
        return candidates

    def graph_matching(self, child):
        pass

def load_model_meta(meta_file='resnet50'):
    print('\n\n')
    start_time = time.time()
    # meta file is rather small
    onnx_model = onnx.load(meta_file+".onnx")
    model_graph = onnx_model.graph

    # record the shape of each weighted nodes
    node_shapes = get_tensor_shapes(model_graph)

    # construct the computation graph and align their attribution
    nodes = onnx_model.graph.node
    graph = igraph.Graph(directed=True)

    node_ids = dict()
    edge_source = collections.defaultdict(list)

    for idx, node in enumerate(nodes):
        input_nodes, trainable_weights = split_inputs(node.input)

        # add new nodes to graph
        attr = {
            'dim': None if not trainable_weights else node_shapes[trainable_weights[0]],
            'op_type': node.op_type,
            'name': node.name,
        }
        graph.add_vertex(name=idx, attr=attr)

        # add edges
        for input_node in input_nodes:
            for s in edge_source[input_node]:
                graph.add_edge(source=s, target=idx)

        # register node 
        for out_node in node.output:
            edge_source[out_node].append(idx)

    #print(graph.vs[2]['attribution']['name'])
    # get_layer_view(graph)
    #print('successors:', graph.successors(node2), graph.out_edges(node2))
    #print(dir(graph.vs[2]))
    opt = MatchingOperator(graph)
    print(opt.get_candidate_store(graph)[3])


    print('\n\nloading takes {} sec \n'.format(time.time() - start_time))

load_model_meta()
