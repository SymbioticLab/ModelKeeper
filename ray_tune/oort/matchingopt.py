import onnx
import numpy
import networkx as nx
import time, sys, os
import functools, collections
from oort.mappingopt import MappingOperator
import logging
from onnx import numpy_helper
import multiprocessing
import torch
import heapq
from multiprocessing import Manager
import ctypes
import json

# Call C backend
clib_matcher = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'backend/bin/matcher.so'))
clib_matcher.get_matching_score.restype = ctypes.c_char_p

sys.setrecursionlimit(10000)
#logging.basicConfig(filename='logging', level=logging.INFO)
INS, DEL, MISMATCH, MATCH = [-2, -1, 0, 1]

def split_inputs(in_list):
    # input list may contain trainable weights
    input_nodes = []
    layer_name = None

    for _input in in_list:
        # tensor nodes are numeric by default
        if _input.isnumeric(): 
            input_nodes.append(_input)
        # in onnx model, weight comes ahead of other trainable weights
        # in some cases, bias itself may be a tensor
        elif '.weight' in _input:
            layer_name = _input
            break
        elif '.bias' in _input:
            layer_name = _input
            break

    return input_nodes, layer_name

def get_tensor_shapes(model_graph):

    node_shapes = dict()
    num_of_trainable_tensors = 0

    if model_graph.initializer:
        #print("Load from initializer")
        for init in model_graph.initializer:
            if '.weight' in init.name:
                num_of_trainable_tensors += 1
            node_shapes[init.name] = tuple(init.dims)
    else:
        #print("Load from input")
        for node in model_graph.input:
            node_shapes[node.name] = tuple([p.dim_value for p in node.type.tensor_type.shape.dim])

    return node_shapes, num_of_trainable_tensors


def clip(var):
    if var < -1.: return -1.
    if var > 1.: return 1.
    return var

def topological_sorting(graph):
    """DFS based topological sort to maximize length of each chain"""
    # return list(nx.topological_sort(graph))
    visited = set()
    ret = []

    def dfs(node):
        visited.add(node)
        [dfs(edge[1]) for edge in graph.out_edges(node) if edge[1] not in visited]
        ret.append(node)

    [dfs(node) for node in graph.nodes() if graph.in_degree(node)==0]
    ret.reverse()
    return ret

class MatchingOperator(object):

    def __init__(self, parent):

        self.parent       = parent
        self.matchidxs  = None
        self.parentidxs    = None
        self.child = None
        self.childidx_order = None

        # parent
        self.nodeIDtoIndex = {}
        self.nodeIndexToID = {-1: None}

        # child
        self.childNodeIDtoIndex = {}

        self.parentidx_order = topological_sorting(self.parent)
        self.parentPrevIndicesList = []

        self.match_res = None
        self.meta_data = {}

        self.init_parent_index()

    def init_parent_index(self):
        # generate a dict of (nodeID) -> (index into nodelist (and thus matrix))
        for (index, nidx) in enumerate(self.parentidx_order):
            self.nodeIDtoIndex[nidx] = index

        # initiate prevs for parent and child graph
        self.parentPrevIndicesList = []
        for (index, nidx) in enumerate(self.parentidx_order):
            self.parentPrevIndicesList.append(self.parentPrevIndices(nidx))

        self.meta_data['parent']= {'opts':[self.parent.nodes[x]['attr']['op_type'] for x in self.parentidx_order],
                                  'dims':[self.parent.nodes[x]['attr']['dims'] for x in self.parentidx_order],
                                  'parents':[self.parentPrevIndicesList[i] for i in range(len(self.parentidx_order))]
                        }

        self.meta_data['len_parent'] = len(self.parentidx_order)

    def align_child(self, child, read_mapping):
        start_time = time.time()

        # reset all parameters
        self.child = child
        self.parentPrevIndicesList = []
        self.childNodeIDtoIndex = {}

        self.childidx_order = topological_sorting(self.child)

        self.init_child_index()

        # call C lib to get mapping
        json_string = self.dump_meta_json()
        ans_json_str = clib_matcher.get_matching_score(ctypes.c_char_p(bytes(json_string, encoding="utf-8")),
                                                       ctypes.c_bool(read_mapping))

        self.match_score, self.match_res = self.parse_json_str(ans_json_str, read_mapping)

        print(f"align_child takes {time.time() - start_time} sec")
        return self.match_score

    def alignmentStrings(self):
        return ("\t".join([self.parent.nodes[j]['attr']['name'] if j is not None else "-" for j in self.parentidxs]), 
                "\t".join([self.child.nodes[i]['attr']['name'] if i is not None else "-" for i in self.matchidxs]),
                self.match_score)

    def graphStrings(self):
        return ("\t".join([self.parent.nodes[j]['attr']['name'] if j is not None else "-" for j in self.parentidx_order]),
                "\t".join([self.child.nodes[i]['attr']['name'] if i is not None else "-" for i in self.childidx_order])
                )

    def get_matching_score(self, child, read_mapping=False):
        score = self.align_child(child=child, read_mapping=read_mapping)
        return score

    def get_mappings(self):

        matches = self.backtrack(*self.match_res)
        self.matchidxs, self.parentidxs = matches

        mapping_res = []
        parent_set, child_set = set(), set()

        ans = self.alignmentStrings()

        for i in range(len(self.parentidxs)):
            if self.parentidxs[i] is not None and self.matchidxs[i] is not None:
                if self.parentidxs[i] not in parent_set and self.matchidxs[i] not in child_set:
                    mapping_res.append((self.parentidxs[i], self.matchidxs[i]))
                    parent_set.add(self.parentidxs[i])
                    child_set.add(self.matchidxs[i])

        return mapping_res, self.match_score

    def dump_meta_json(self):
        #start_time = time.time()
        self.meta_data['len_child'] = len(self.childidx_order)
        self.meta_data['child'] = {'opts':[self.child.nodes[x]['attr']['op_type'] for x in self.childidx_order],
                                   'dims':[self.child.nodes[x]['attr']['dims'] for x in self.childidx_order],
                                   'parents':[self.childPrevIndicesList[x]  for x in self.childidx_order]}

        json_str = json.dumps(self.meta_data)

        return json_str

    def parse_json_str(self, ans_json_str, read_mapping):
        ans = json.loads(ans_json_str)

        score = ans['score']
        matches = None

        if read_mapping:
            backGrphIdx, backStrIdx = collections.defaultdict(list), collections.defaultdict(list)

            # parse information
            for key in ans['backParentIdx']:
                hash_v = key.split('_') 
                backGrphIdx[int(hash_v[0]), int(hash_v[1])] = ans['backParentIdx'][key]

            for key in ans['backChildIdx']:
                hash_v = key.split('_') 
                backStrIdx[int(hash_v[0]), int(hash_v[1])] = ans['backChildIdx'][key]

            matches = [backGrphIdx, backStrIdx]

        return score, matches

    def parentPrevIndices(self, node):
        """Return a list of the previous dynamic programming table indices
           corresponding to predecessors of the current node."""
        prev = []
        for source, target in self.parent.in_edges(node):
            prev.append(self.nodeIDtoIndex[source])

        # if no predecessors, point to just before the parent
        if len(prev) == 0:
            prev = [-1]
        return prev


    def childPrevIndices(self, node):
        prev = []
        for source, target in self.child.in_edges(node):
            prev.append(self.childNodeIDtoIndex[source])
        if len(prev) == 0:
            prev = [-1]
        return prev


    def init_child_index(self):

        for (index, nidx) in enumerate(self.childidx_order):
            self.childNodeIDtoIndex[nidx] = index

        self.childPrevIndicesList = {}
        for (index, nidx) in enumerate(self.childidx_order):
            self.childPrevIndicesList[nidx] = self.childPrevIndices(nidx)


    def backtrack(self, backGrphIdx, backStrIdx):
        """Backtrack through the scores and backtrack arrays.
           Return a list of child indices and node IDs (not indices, which
           depend on ordering)."""
        besti, bestj = len(self.parentidx_order), len(self.childidx_order)

        matches, strindexes = [], []
        que = [(besti, bestj)]
        que_set = set([(besti, bestj)])

        while len(que) != 0:
            besti, bestj = que.pop()
            curstridx, curnodeidx = self.childidx_order[bestj-1], self.parentidx_order[besti-1]

            nextis, nextjs = backGrphIdx[besti, bestj], backStrIdx[besti, bestj] # last step to (besti, bestj)

            name_aligned = True
            if curstridx is not None and curnodeidx is not None:
                name_aligned = self.child.nodes[curstridx]['attr']['op_type']==self.parent.nodes[curnodeidx]['attr']['op_type']

            # we pad a gap
            name_aligned = name_aligned and (bestj not in nextjs and besti not in nextis) 

            strindexes.append(curstridx if name_aligned else None)
            matches.append(curnodeidx if name_aligned else None)

            for nexti, nextj in zip(nextis, nextjs):
                if not(nexti == 0 and nextj == 0) and (nexti, nextj) not in que_set:
                    que.append((nexti, nextj)) # DFS
                    que_set.add((nexti, nextj))

        strindexes.reverse()
        matches.reverse()

        return strindexes, matches


def mapping_func(parent_opt, child, read_mapping=False):
    score = parent_opt.get_matching_score(child=child, read_mapping=read_mapping)
    return (parent_opt.parent.graph['name'], score) #(self.model_zoo[parent_path].parent, mappings, score)


class Oort(object):

    def __init__(self, args):
        self.args = args

        self.model_zoo = collections.OrderedDict()

        if args.zoo_path is not None:
            self.init_model_zoo(args.zoo_path)

        self.current_mapping_id = 0
        self.skip_opts = {'Constant'}

    def init_model_zoo(self, zoo_path):
        if '.onnx' in zoo_path: 
            model_paths = [zoo_path]
        else:
            model_paths = [os.path.join(zoo_path, x) for x in os.listdir(zoo_path) \
                if os.path.isfile(os.path.join(zoo_path, x)) and '.onnx' in x]

        for idx, model_path in enumerate(model_paths):
            self.add_to_zoo(model_path, idx)
            print(f"Added {model_path} to zoo ...")


    def load_model_meta(self, meta_file='sample.onnx'):
        skip_opts = {'Constant'}
        start_time = time.time()
        # meta file is rather small
        onnx_model = onnx.load(meta_file)
        model_graph = onnx_model.graph

        # record the shape of each weighted nodes
        node_shapes, num_of_trainable_tensors = get_tensor_shapes(model_graph)

        # construct the computation graph and align their attribution
        nodes = [n for n in onnx_model.graph.node if n.op_type not in skip_opts]
        graph = nx.DiGraph(name=meta_file, num_tensors=num_of_trainable_tensors)

        node_ids = dict()
        edge_source = collections.defaultdict(list)

        opt_dir = collections.defaultdict(int)
        for idx, node in enumerate(nodes):
            input_nodes, trainable_weights = split_inputs(node.input)
            opt_dir[node.op_type] += 1

            #print(node.input, trainable_weights)
            # add new nodes to graph
            attr = {
                'dims': [] if not trainable_weights else node_shapes[trainable_weights],
                'op_type': node.op_type,
                'name': node.name,# if node.name else str(node.op_type)+str(opt_dir[node.op_type]),
                'layer_name': None if not trainable_weights else '.'.join(trainable_weights.split('.')[:-1])
            }
            graph.add_node(idx, attr=attr)

            # add edges
            for input_node in input_nodes:
                for s in edge_source[input_node]:
                    graph.add_edge(s, idx)

            # register node 
            for out_node in node.output:
                edge_source[out_node].append(idx)

        return graph, onnx_model


    def add_to_zoo(self, model_path, idx):
        try:
            model_graph, model_weight = self.load_model_meta(model_path)
            model_graph.graph['model_id'] = str(idx)
            self.model_zoo[model_path] = MatchingOperator(parent=model_graph)
        except Exception as e:
            print(f"Error: {e} for {model_path}")


    def remove_from_zoo(self, model_path):
        if model_path in self.model_zoo:
            del self.model_zoo[model_path]
        else:
            logging.warning(f"Fail to remove {model_path} from zoo, as it does not exist")


    def get_mappings(self, parent_path):
        mapping_res, score = self.model_zoo[parent_path].get_mappings()
        return (self.model_zoo[parent_path].parent, mapping_res, score)


    def get_best_mapping(self, child, blacklist=set(), model_name=None):

        start_time = time.time()

        pool = multiprocessing.Pool(processes=self.args.num_of_processes)
        results = []

        for model_path in self.model_zoo.keys():
            if model_path not in blacklist:
                results.append(pool.apply_async(mapping_func, (self.model_zoo[model_path], child)))

        pool.close()
        pool.join()

        parent_path, mappings, best_score = None, [], float('-inf')
        parent = None
        
        for res in results:
            (p, s) = res.get()
            #logging.info(f"For mapping pair ({model_name}, {p.graph['name']}) score is {s}")
            print(f"For mapping pair ({model_name}, {p}) score is {s}")
            if s > best_score:
                parent_path, best_score = p, s

        if parent_path is not None:
            mapping_func(self.model_zoo[parent_path], child, read_mapping=True)
            parent, mappings, best_score = self.get_mappings(parent_path)

        print("takes {:.2f} sec".format(time.time() - start_time))
        if parent is not None:
            print("Find best mappings {} (score: {}) takes {:.2f} sec\n\n".format(
                                parent.graph['name'], best_score, time.time() - start_time))

        return parent, mappings, best_score


    def warm_weights(self, parent, child, mappings):

        mapper = MappingOperator(parent, child, mappings)
        mapper.cascading_mapping()
        mapper.pad_mapping()
        weights, num_of_matched = mapper.get_mapping_weights()

        print("Mapped {} layers to the child model ({} layers), parent {} layers".format(num_of_matched, child.graph['num_tensors'],
                    parent.graph['num_tensors']))

        return weights, num_of_matched



    def map_for_model(self, child_model, dummy_input, hidden = None, blacklist=set(), model_name=None):
        """
        @ child_model: model to warm start
        @ dummpy_input: randomly generated input to infer the shape of model
        @ blacklist: blacklist certain models in the zoo during matching
        """

        self.current_mapping_id += 1

        # dump the model into onnx format
        onnx_model_name = os.path.join(self.args.exe_path, str(self.current_mapping_id)+".onnx_temp")
        if hidden is None:
            torch.onnx.export(child_model, dummy_input, onnx_model_name, 
                        export_params=True, verbose=0, training=1, do_constant_folding=False)
        else:
            with torch.no_grad():
                output, hidden = child_model(dummy_input, hidden)
                torch.onnx.export(child_model, (dummy_input, hidden), onnx_model_name,
                            export_params=True, verbose=0, training=1, 
                            do_constant_folding=False, 
                            input_names=['dummy_input'],
                            output_names=['output'],
                            dynamic_axes={'dummy_input': [0], 'output': [0]})


        child, child_onnx = self.load_model_meta(onnx_model_name)
        child.graph['model_id'] = str(self.current_mapping_id)

        # find the best mapping from the zoo
        parent, mappings, best_score = self.get_best_mapping(child, blacklist, model_name)

        # overwrite the current model weights
        weights, num_of_matched = None, 0
        parent_name = 'None'

        if parent is not None:
            weights, num_of_matched = self.warm_weights(parent, child, mappings)
            parent_name = parent.graph['name']

        # remove the temporary onnx model
        os.remove(onnx_model_name)

        return weights, num_of_matched, parent_name

    def map_for_onnx(self, child_onnx_path, blacklist=set(), model_name=None):
        child, child_onnx = self.load_model_meta(child_onnx_path)
        child.graph['model_id'] = str(self.current_mapping_id)

        # find the best mapping from the zoo
        parent, mappings, best_score = self.get_best_mapping(child, blacklist, model_name)

        # overwrite the current model weights
        weights, num_of_matched = None, 0
        parent_name = 'None'

        if parent is not None:
            weights, num_of_matched = self.warm_weights(parent, child, mappings)
            parent_name = parent.graph['name']

        return weights, num_of_matched, parent_name

def faked_graph():
    graph = nx.DiGraph(name='faked')
    attr1={'dims': [64, 32, 3, 3], 'op_type': 'cov1',}

    graph.add_node(0, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'0'})

    graph.add_node(1, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'1'})
    graph.add_node(2, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'2'})

    #graph.add_node(5, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'5'})
    #graph.add_node(3, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'3'})

    graph.add_node(4, attr={'dims': [5, 32, 3, 3], 'op_type': 'cov1', 'name':'4'})

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 4)
    #graph.add_edge(0, 5)
    #graph.add_edge(5, 3)
    #graph.add_edge(3, 4)

    return graph

def faked_graph2():
    graph = nx.DiGraph(name='faked')
    attr1={'dims': [64, 32, 3, 3], 'op_type': 'cov1',}

    graph.add_node(0, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'0'})

    graph.add_node(1, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'1'})
    graph.add_node(2, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'2'})

    graph.add_node(5, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'5'})
    graph.add_node(3, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'3'})

    graph.add_node(4, attr={'dims': [5, 32, 3, 3], 'op_type': 'cov1', 'name':'4'})

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 4)
    graph.add_edge(0, 5)
    graph.add_edge(5, 3)
    graph.add_edge(3, 4)

    return graph


def mapping_faked(parent, child_graph):
    opt = MatchingOperator(parent=parent)
    mappings, score = opt.get_mappings(child=child_graph)

    print(opt.alignmentStrings()[0])
    print("\n\n")
    print(opt.alignmentStrings()[1])
    print("\n\n")
    print(opt.graphStrings()[0])
    print("\n\n")
    print(opt.graphStrings()[1])
    return (parent, mappings, score)

def test_fake():
    parent, child = faked_graph(), faked_graph2()

    mapper = MatchingOperator(parent=parent)
    print(mapper.get_mappings(child))


def test():
    import argparse

    start_time = time.time()
    zoo_path = '/gpfs/gpfs0/groups/chowdhury/fanlai/model_zoo/imagenet120/500_800/cos_lr/random/'
    #zoo_path = '/gpfs/gpfs0/groups/chowdhury/fanlai/net_transformer/Net2Net/torchzoo/shufflenet_v2_x2_0.onnx'

    parser = argparse.ArgumentParser()
    parser.add_argument('--zoo_path', type=str, default=zoo_path)
    parser.add_argument('--num_of_processes', type=int, default=30)
        
    args = parser.parse_args()

    mapper = Oort(args)

    child_onnx_path = '/gpfs/gpfs0/groups/chowdhury/fanlai/model_zoo/imagenet120/500_800/cos_lr/random/model_134.pth.onnx'
    #child_onnx_path = '/gpfs/gpfs0/groups/chowdhury/fanlai/net_transformer/Net2Net/torchzoo/shufflenet_v2_x2_0.onnx'
    weights, num_of_matched, parent_name = mapper.map_for_onnx(child_onnx_path, blacklist=set([]))


    print("\n\nMatched {} layers".format(num_of_matched))


#test()
#test_fake()


