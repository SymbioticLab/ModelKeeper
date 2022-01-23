import onnx
import networkx as nx
import time, sys, os
import functools, collections
import logging
from onnx import numpy_helper
import multiprocessing
import torch
import concurrent.futures


import ctypes
import json
import concurrent, threading
import pickle
import gc
from itertools import repeat
import shutil
import random
import numpy as np
import bisect

sys.path.append('./modelkeeper')

from mappingopt import MappingOperator
# Libs for model clustering
from clustering import k_medoids
from evictor import mip

# AED
from aed_matcher import AEDMatcher

# Call C backend
clib_matcher = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'backend/bin/matcher.so'))
clib_matcher.get_matching_score.restype = ctypes.c_char_p

sys.setrecursionlimit(10000)
random.seed(1)
distance_lookup = None
SCORE_THRESHOLD = float('-inf')
THRESHOLD = 0.1 # more than X% layers can be transferred from the parent
MAX_MATCH_NODES=5000
HIT_BENEFIT=1.0
IS_AED = False
AED_PATH = None

log_path = './modelkeeper_log'
with open(log_path, 'w') as fout:
    pass

logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)s %(message)s',
                datefmt='%H:%M:%S',
                level=logging.INFO,
                handlers=[
                    logging.FileHandler(log_path, mode='a'),
                    logging.StreamHandler()
                ])


def get_distance(a, b):
    return distance_lookup[a][b] if a != b else 0.

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
            # break
        elif layer_name is None and '.bias' in _input:
            layer_name = _input
            # break

    # if in_list != input_nodes:
    #     print(in_list, input_nodes)
    return input_nodes, layer_name

def get_tensor_shapes(model_graph):

    node_shapes = dict()
    num_of_trainable_tensors = 0

    if model_graph.initializer:
        #logging.info("Load from initializer")
        for init in model_graph.initializer:
            if '.weight' in init.name:
                num_of_trainable_tensors += 1
            node_shapes[init.name] = tuple(init.dims)
    else:
        #logging.info("Load from input")
        for node in model_graph.input:
            node_shapes[node.name] = tuple([p.dim_value for p in node.type.tensor_type.shape.dim])

    return node_shapes, num_of_trainable_tensors


def topological_sorting(graph):
    """DFS based topological sort to maximize length of each chain"""
    ret = []
    in_degrees = {n:graph.in_degree(n) for n in graph.nodes if graph.in_degree(n) > 0}

    def dfs_iterative(start_vertex):
        stack = [start_vertex]

        while stack:
            vertex = stack.pop()
            ret.append(vertex)

            temp_out = []
            for edge in graph.out_edges(vertex):
                if in_degrees[edge[1]] == 1:
                    temp_out.append(edge[1])
                    del in_degrees[edge[1]]
                else:
                    in_degrees[edge[1]] -= 1

            stack += temp_out

    [dfs_iterative(node) for node in graph.nodes() if graph.in_degree(node)==0]
    assert len(ret) == graph.number_of_nodes()

    #for n in ret:
    #    print(n, graph.nodes[n]['attr'])
    #for node in ret:
        #print(graph.nodes[node]['attr']['layer_name'])
    #print('** Source: ', sum([1 for node in graph.nodes() if graph.in_degree(node) == 0]))
    #print('** Sink: ', sum([1 for node in graph.nodes() if graph.out_degree(node) == 0]))
    return ret[:MAX_MATCH_NODES]


def get_onnx_name(file):
    return file.split('/')[-1].split('@')[0]

class MatchingOperator(object):

    def __init__(self, parent):

        self.parent = parent
        self.matchidxs = None
        self.parentidxs = None
        self.child = None
        self.childidx_order = None

        # parent
        self.nodeIDtoIndex = {}
        self.nodeIndexToID = {-1: None}

        # child
        self.childNodeIDtoIndex = {}

        self.parentidx_order = topological_sorting(self.parent)
        self.parentPrevIndicesList = {}

        self.match_res = None
        self.meta_data = {}

        self.num_of_nodes = len(self.parentidx_order)

        self.init_parent_index()

        # for eviction
        self.model_value = 1.
        self.model_weight = 1.

        global IS_AED, AED_PATH
        self.aed_matcher = AEDMatcher(AED_PATH, get_onnx_name(parent.graph['name'])) if IS_AED else None

    def init_parent_index(self):
        # generate a dict of (nodeID) -> (index into nodelist (and thus matrix))
        for (index, nidx) in enumerate(self.parentidx_order):
            self.nodeIDtoIndex[nidx] = index

        # initiate prevs for parent and child graph
        self.parentPrevIndicesList = {}
        for (index, nidx) in enumerate(self.parentidx_order):
            self.parentPrevIndicesList[nidx] = self.parentPrevIndices(nidx)

        self.meta_data['parent']= {'opts':[self.parent.nodes[x]['attr']['op_type'] for x in self.parentidx_order],
                                  'dims':[self.parent.nodes[x]['attr']['dims'] for x in self.parentidx_order],
                                  'parents':[self.parentPrevIndicesList[x] for x in self.parentidx_order]
                        }

        self.meta_data['len_parent'] = len(self.parentidx_order)


    def align_child(self, child, read_mapping):

        start_time = time.time()

        # reset all parameters
        self.child = child
        self.childNodeIDtoIndex = {}

        self.childidx_order = topological_sorting(self.child)

        self.init_child_index()

        # AED Matcher will directly read the storage
        if self.aed_matcher is not None:
            self.match_score, self.match_res = self.aed_matcher.query_child(get_onnx_name(child.graph['name']))
            return self.match_score

        # call C lib to get mapping
        json_string = self.dump_meta_json()
        ans_json_str = clib_matcher.get_matching_score(ctypes.c_char_p(bytes(json_string, encoding="utf-8")),
                                                       ctypes.c_bool(read_mapping))

        self.match_score, self.match_res = self.parse_json_str(ans_json_str, read_mapping)

        logging.info(f"{self.parent.graph['name'].split('/')[-1]} align_child {self.child.graph['name'].split('/')[-1]} takes {time.time() - start_time} sec, " +
                f"score: {round(self.match_score/len(self.childidx_order), 4)}")

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
        return score/len(self.childidx_order)# child.graph['num_tensors']

    def get_mappings(self):
        # AED Matcher will directly read the storage
        if self.aed_matcher is not None:
            self.match_score, self.match_res = self.aed_matcher.query_child(get_onnx_name(self.child.graph['name']))
            return self.match_res, self.match_score

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
                                   'parents':[self.childPrevIndicesList[x] for x in self.childidx_order]}#,
                                   ##'out_degrees':[self.child.out_degree(x) for x in self.childidx_order]}
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

    def increase_value(self, new_v):
        self.model_value += new_v

    def update_weight(self, new_w):
        self.model_weight = new_w


def mapping_func(parent_opt, child, read_mapping=False, return_child_name=False):
    parent_name = parent_opt.parent.graph['name']

    try:
        score = parent_opt.get_matching_score(child=child, read_mapping=read_mapping)
    except Exception as e:
        logging.error(f"Mapping {parent_name} to {child.graph['name']} failed as {e}")
        score = float('-inf')

    if not read_mapping:
        if return_child_name:
            return parent_name, score, child.graph['name']
        return parent_name, score
    else:
        mapping_res, score = parent_opt.get_mappings()
        return (parent_opt.parent, mapping_res, score)

class ModelKeeper(object):

    def __init__(self, args):
        self.args = args

        self.model_zoo = collections.OrderedDict()

        self.current_mapping_id = 0
        self.zoo_model_id = 0
        self.mode_threshold = 1000 # enable clustering if len(model_zoo) > T

        self.model_clusters = []
        self.query_model = None

        # Storage of the model pair distance: {parent_model: {child_model}}
        self.distance = collections.defaultdict(dict)

        # Lock for offline clustering
        # self.zoo_lock = threading.Lock()

        # Blacklist opts in matching
        self.skip_opts = {'Constant'}
        self.opt_annotation = set(['key', 'value', 'query'])
        self.outlier_factor = 3.0
        self.zoo_storage = 0.
        self.VALUE_DECAY_FACTOR = 0.99
        self.zoo_capacity = self.args.zoo_capacity

        # Bucketing model selection
        self.bucket_selection = args.bucketing_selection
        self.bucket_interval = args.bucket_interval

        if args.aed_match:
            global IS_AED, AED_PATH
            IS_AED = args.aed_match
            AED_PATH = args.aed_path

        if args.zoo_path is not None:
            self.init_model_zoo(args.zoo_path)

        self.init_execution_store()
        self.service_thread = None


    def init_model_zoo(self, zoo_path):
        if os.path.exists(zoo_path):
            model_paths = [os.path.join(zoo_path, x) for x in os.listdir(zoo_path) if x.endswith('.onnx')]

            self.add_to_zoo(model_paths)

    def init_execution_store(self):

        runtime_stores = [self.args.zoo_path, self.args.zoo_query_path,
                            self.args.zoo_ans_path, self.args.zoo_register_path]

        for store in runtime_stores:
            if not os.path.exists(store):
                os.makedirs(store)

    def clean_execution_store(self):
        runtime_stores = [self.args.zoo_path, self.args.zoo_query_path,
                            self.args.zoo_ans_path, self.args.zoo_register_path]

        for store in runtime_stores:
            if os.path.exists(store):
                shutil.rmtree(store)


    def add_to_zoo(self, model_paths):
        """Register new model to the existing zoo"""
        if not isinstance(model_paths, list):
            model_paths = [model_paths]

        if len(model_paths) == 0:
            return

        model_accuracies = [p.parent.graph['accuracy'] for p in self.model_zoo.values()]
        new_model_accuracies = []
        for m in model_paths:
            model_acc = self.get_model_accuracy(m)
            new_model_accuracies.append(model_acc)

        # remove outliers
        model_accuracies = np.array(model_accuracies+new_model_accuracies)
        accuracy_std, accuracy_mean = model_accuracies.std(), model_accuracies.mean()

        existing_zoo = list(self.model_zoo.keys())
        for m in existing_zoo:
            if self.model_zoo[m].parent.graph['accuracy'] < accuracy_mean - self.outlier_factor*accuracy_std:
                del self.model_zoo[m]

        decent_models = []
        for acc, m in zip(new_model_accuracies, model_paths):
            if acc >= accuracy_mean - self.outlier_factor*accuracy_std:
                decent_models.append(m)

        # decay model value by factor
        for m in self.model_zoo:
            self.model_zoo[m].model_value *= self.VALUE_DECAY_FACTOR

        is_update = False
        for model_path in decent_models:
            logging.info(f"Try to add {model_path} to zoo ...")
            if model_path in self.model_zoo:
                logging.warning(f"{model_path} is already in the zoo")
            else:
                try:
                    model_graph, model_weight = self.load_model_meta(model_path)
                    model_graph.graph['model_id'] = str(self.zoo_model_id)

                    model_size = os.path.getsize(model_path)/1024./1024. # MB
                    #with self.zoo_lock:
                    self.model_zoo[model_path] = MatchingOperator(parent=model_graph)
                    self.model_zoo[model_path].update_weight(model_size) # MB
                    self.zoo_model_id += 1
                    self.zoo_storage += model_size
                    is_update = True

                    logging.info(f"Added {model_path} to zoo ...")
                except Exception as e:
                    logging.info(f"Error: {e} for {model_path}")

        # Evict models to cap zoo size
        if is_update and self.zoo_storage > self.zoo_capacity:
            logging.info(f"Try to evict model to cap size")

            total_util, models_to_evict = self.get_models_evict()
            size_before_evict = int(self.zoo_storage)
            util_before_evict = int(sum([self.model_zoo[m].model_value for m in self.model_zoo]))

            logging.info(f"Evict model {[(self.model_zoo[m].model_weight, m) for m in models_to_evict]}")

            self.remove_from_zoo(models_to_evict)

            logging.info(f"Zoo storage ({size_before_evict} MB, {util_before_evict}) updates to ({int(self.zoo_storage)} MB,"+
                f" {total_util}), target ({self.zoo_capacity} MB), {len(self.model_zoo)} models left")

        # Update model zoo
        if is_update and self.mode_threshold < len(self.model_zoo):
            self.update_model_clusters()


    def get_models_evict(self):

        """Solve the knapsack problem to pick models"""

        model_names = list(self.model_zoo.keys())
        weight_list = [self.model_zoo[m].model_weight for m in model_names]
        util_list = [self.model_zoo[m].model_value for m in model_names]

        logging.info(weight_list)
        total_util, res = mip(self.zoo_capacity, weight_list, util_list)
        evict_models = [model_names[i] for i in range(len(res)) if res[i] == 0]
        logging.info(res)

        return int(total_util), evict_models


    def remove_from_zoo(self, model_paths):
        is_update = False

        if not isinstance(model_paths, list):
            model_paths = [model_paths]

        for model_path in model_paths:
            if model_path in self.model_zoo:
                #with self.zoo_lock:
                self.zoo_storage -= self.model_zoo[model_path].model_weight
                del self.model_zoo[model_path]
                is_update = True
            else:
                logging.warning(f"Fail to remove {model_path} from zoo, as it does not exist")

        # Update model zoo asyn
        if is_update and self.mode_threshold < len(self.model_zoo):
            self.update_model_clusters()


    def evict_neighbors(self, models_for_clustering):
        """
            @ models_for_clustering: models considered as parent candidates
        """
        evicted_nodes = set()

        # Scores are not symmetric
        for model_a in models_for_clustering:
            for model_b in models_for_clustering:
                if model_a not in evicted_nodes and model_b not in evicted_nodes:
                    if model_a != model_b and self.distance[model_a][model_b] < self.args.neigh_threshold and \
                        self.distance[model_b][model_a] < self.args.neigh_threshold:
                        # evict the one w/ lower accuracy
                        if self.model_zoo[model_a].parent.graph['accuracy'] != -1 and self.model_zoo[model_b].parent.graph['accuracy'] != -1:
                            if self.model_zoo[model_a].parent.graph['accuracy'] < self.model_zoo[model_b].parent.graph['accuracy']:
                                evicted_nodes.add(model_a)
                            else:
                                evicted_nodes.add(model_b)

        logging.info(f"Evicting {len(evicted_nodes)} from zoo")

        return [n for n in models_for_clustering if n not in evicted_nodes]


    def update_model_clusters(self, threads=40, num_of_clusters=None, spawn=10000):
        """
            @ threads: number of threads to simulate {spawn} clustering trials
        """
        logging.info(f"Clustering {len(self.model_zoo)} models ...")

        #with self.zoo_lock:
        current_zoo_models = list(self.model_zoo.keys())

        def update_cluster_offline():
            global distance_lookup
            nonlocal num_of_clusters, spawn, threads, current_zoo_models

            # 1. Update all pairwise distance offline
            updating_pairs = [(i, j) for i in current_zoo_models for j in current_zoo_models
                            if (i not in self.distance or j not in self.distance[i]) and i != j]

            # Give a chance that every model has some edges
            random.shuffle(updating_pairs)
            pool = multiprocessing.Pool(processes=threads)
            results = []

            for (model_a, model_b) in updating_pairs:
                results.append(pool.apply_async(mapping_func,
                        (self.model_zoo[model_a], self.model_zoo[model_b].parent, False, True)))

            pool.close()
            pool.join()

            for res in results:
                parent_name, transfer_score, child_name = res.get()
                # Dict is thread-safe in Python
                self.distance[parent_name][child_name] = 1. - transfer_score

            # Pickle in threads can not support nested function
            distance_lookup = self.distance

            #  2. Evict neighbors if similarity > threshold (0.95 default) and neigh_acc < node_acc
            current_zoo_models = self.evict_neighbors(current_zoo_models)

            if num_of_clusters is None:
                num_of_clusters = int(len(current_zoo_models)**0.5)

            diameter, self.model_clusters = k_medoids(current_zoo_models,
                            k=num_of_clusters, distance=get_distance,
                            threads=threads, spawn=spawn, max_iterations=1000)

            logging.info(f"Cluster into {num_of_clusters} clusters, max_diameter: {diameter}")

        thread = threading.Thread(target=update_cluster_offline)
        thread.start()
        #update_cluster_offline()


    def get_op_type(self, op_type, node_name):
        temp_op = op_type
        if node_name:
            for annotation in self.opt_annotation:
                if annotation in node_name:
                    temp_op=temp_op+'_'+annotation

        return temp_op

    def get_model_accuracy(self, meta_file):
        if '@' in meta_file:
            accuracy = float(meta_file.split('@')[-1].split('.onnx')[0])
        else:
            accuracy = -1
        return accuracy

    def load_model_meta(self, meta_file='sample__accuracy.onnx'):
        """
        @ meta_file: input files are onnx. return the weight meta graph of this model
        """

        start_time = time.time()

        # meta file is rather small
        onnx_model = onnx.load(meta_file)
        model_graph = onnx_model.graph
        accuracy = self.get_model_accuracy(meta_file)

        # record the shape of each weighted nodes
        node_shapes, num_of_trainable_tensors = get_tensor_shapes(model_graph)

        # construct the computation graph and align their attribution
        nodes = [n for n in onnx_model.graph.node if n.op_type not in self.skip_opts]
        graph = nx.DiGraph(name=meta_file, num_tensors=num_of_trainable_tensors, accuracy=accuracy,
                            num_nodes=len(nodes))

        node_ids = dict()
        edge_source = collections.defaultdict(list)

        opt_dir = collections.defaultdict(int)
        input_nodes_list = []
        for idx, node in enumerate(nodes):
            input_nodes, trainable_weights = split_inputs(node.input)
            opt_dir[node.op_type] += 1

            #logging.info(node.input, trainable_weights)
            # add new nodes to graph
            layer_name = None if not trainable_weights else '.'.join(trainable_weights.split('.')[:-1])

            attr = {
                'dims': [] if not trainable_weights else node_shapes[trainable_weights],
                'op_type': self.get_op_type(node.op_type, layer_name),
                'name': node.name,# if node.name else str(node.op_type)+str(opt_dir[node.op_type]),
                'layer_name': layer_name
            }
            graph.add_node(idx, attr=attr)

            # register node
            for out_node in node.output:
                edge_source[out_node].append(idx)

            input_nodes_list.append(input_nodes)


        for idx, node in enumerate(nodes):
            input_nodes = input_nodes_list[idx]

            # add edges
            for input_node in input_nodes:
                for s in edge_source[input_node]:
                    graph.add_edge(s, idx)
        # import matplotlib.pyplot as plt
        # nx.draw_spectral(graph,node_size=10, arrowsize=5)
        # plt.savefig(f"{graph.graph['name']}.pdf")
        return graph, onnx_model


    def get_mappings(self, parent_path):
        mapping_res, score = self.model_zoo[parent_path].get_mappings()
        return (self.model_zoo[parent_path].parent, mapping_res, score)


    def query_scores(self, parents, child, threads=20, timeout=600):
        scores = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            try:
                for model, score in executor.map(mapping_func, parents, repeat(child), timeout=timeout):
                    scores.append((model, score))
            except Exception as e:
                for pid, process in executor._processes.items():
                    process.terminate()
                logging.warning(f"Query scores for {child.graph['name']} fails, as: {e}")

        for (p, s) in scores:
            self.distance[p][child.graph['name']] = 1. - s

        return scores

    def query_best_mapping(self, child, blacklist=set(),
                            model_name=None, return_weight=True,
                            score_threshold=0.95, timeout=600, threads=20):

        start_time = time.time()
        self.query_model = child

        # 1. Pick top-k clusters
        medoids = [medoid.kernel for medoid in self.model_clusters]
        medoid_dist = []

        if len(medoids) > 0:
            medoid_dist = self.query_scores([self.model_zoo[p] for p in medoids], child, self.args.num_of_processes)
            #self.model_clusters.sort(key=lambda k:medoid_dist[k.kernel][1], reverse=True)
            medoid_dist.sort(key=lambda k:k[1], reverse=True)

        best_score = SCORE_THRESHOLD
        search_models = []
        parent_path = mappings = parent = None

        # 2. Search members inside top-k clusters in order
        for top_i in range(len(medoid_dist)):
            cluster, score = medoid_dist[top_i]
            for cluster_id in self.model_clusters:
                if cluster_id.kernel == cluster:
                    search_models += [self.model_zoo[e] for e in self.model_clusters[top_i].elements if e != cluster]
                    break

            if score > best_score:
                parent_path, best_score = cluster, score

            self.distance[cluster][child.graph['name']] = 1.0-score

        logging.info(f"Searching {len(search_models)+len(medoids)} models ...")

        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            #for model, score in executor.map(self.mapping_func, search_models, timeout=timeout):
            try:
                for model, score in executor.map(mapping_func, search_models, repeat(child), timeout=timeout):
                    if score > best_score:
                        parent_path, best_score = model, score

                    self.distance[model][child.graph['name']] = 1.0-score

                    if best_score >= score_threshold:
                        executor.shutdown(wait=False)
                        break
            except:
                for pid, process in executor._processes.items():
                    process.terminate()

        if parent_path is not None and return_weight:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                try:
                    for p, m, s in executor.map(mapping_func, [self.model_zoo[parent_path]], [child], [True], timeout=timeout):
                        parent, mappings, _ = p, m, s
                        break
                except Exception as e:
                    logging.warning(f"Query scores for {child.graph['name']} fails, as: {e}")

        if parent is not None:
            logging.info("{} find best mappings {} (score: {}) takes {:.2f} sec\n\n".format(
                        child.graph['name'], parent.graph['name'], round(best_score,4), time.time() - start_time))
        else:
            logging.info("{} does not find best mapping, takes {:.2f} sec\n\n".format(
                        child.graph['name'], time.time() - start_time))

        return parent, mappings, best_score

    def bucketing_selection(self, results, _max, _min):
        if len(results) == 0:
            return None, None

        score_range = _max - _min + 1e-4

        bucket_boundary = [i*1./self.bucket_interval for i in range(self.bucket_interval)]
        buckets = [[] for _ in range(self.bucket_interval)]

        for (p, s) in results:
            norm_score = (s-_min)/score_range
            bucket_id = min(bisect.bisect_left(bucket_boundary, norm_score), self.bucket_interval-1)
            buckets[bucket_id].append((p, s, self.model_zoo[p].parent.graph['accuracy']))

        logging.info(f"bucketing information:\n{buckets}\n, max {_max}, min {_min}")
        for i in range(self.bucket_interval-1, -1, -1):
            if len(buckets[i]) > 0:
                (p, s, a) = max(buckets[i], key=lambda k:k[-1])
                return p, s

        return None, None

    def get_best_mapping(self, child, blacklist=set(), model_name=None, return_weight=True, timeout=600):
        """
            Enumerate all possible model pairs. Not as efficient as the clustering one.
        """
        start_time = time.time()
        self.query_model = child

        parent_models = [model for model in self.model_zoo.keys() if model not in blacklist]
        results = self.query_scores([self.model_zoo[p] for p in parent_models]+[MatchingOperator(child)], child, self.args.num_of_processes)

        parent_path = mappings = parent = None
        best_score, self_score = SCORE_THRESHOLD, 1
        worst_score = float('inf')

        matching_results = []
        for (p, s) in results:
            #logging.info(f"For mapping pair ({model_name}, {p.graph['name']}) score is {s}")
            logging.info(f"For mapping pair ({model_name}, {p}) score is {s}")
            if p != child.graph['name']:
                if s > best_score:
                    parent_path, best_score = p, s
                if s < worst_score:
                    worst_score = s
                matching_results.append((p, s))
            else:
                self_score = s

        if self.bucket_selection == True:
            parent_path, best_score = self.bucketing_selection(matching_results, self_score, worst_score)

        if parent_path is not None and return_weight:
            with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
                try:
                    for p, m, s in executor.map(mapping_func, [self.model_zoo[parent_path]], [child], [True], timeout=timeout):
                        parent, mappings, _ = p, m, s
                        break
                except Exception as e:
                    for pid, process in executor._processes.items():
                        process.terminate()
                    logging.warning(f"Query scores for {child.graph['name']} fails, as: {e}")

        if parent is not None:
            logging.info("{} find best mappings {} (score: {}, {}) takes {:.2f} sec\n\n".format(
                        child.graph['name'], parent.graph['name'], round(best_score,4), round(best_score/self_score, 4), time.time() - start_time))
        else:
            logging.info("{} does not find best mapping, takes {:.2f} sec\n\n".format(
                        child.graph['name'], time.time() - start_time))

        return parent, mappings, best_score


    def warm_weights(self, parent, child, mappings):

        mapper = MappingOperator(parent, child, mappings)
        mapper.cascading_mapping()
        mapper.pad_mapping()
        weights, num_of_matched, layer_mappings = mapper.get_mapping_weights()

        logging.info("Mapped {} layers to the child model ({} layers), parent {} layers".format(num_of_matched, child.graph['num_tensors'],
                    parent.graph['num_tensors']))

        return weights, num_of_matched, layer_mappings



    def map_for_model(self, child_model, dummy_input, hidden = None, blacklist=set(), model_name=None):
        """
        @ child_model: model to warm start
        @ dummpy_input: randomly generated input to infer the shape of model
        @ blacklist: blacklist certain models in the zoo during matching
        """

        self.current_mapping_id += 1

        # dump the model into onnx format
        onnx_model_name = os.path.join(self.args.zoo_query_path, str(self.current_mapping_id)+".onnx_temp")
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
        parent_name, meta_data = 'None', {}

        if mappings is not None and len(mappings) > 0:
            logging.info(f"{sorted(mappings)}\n# of parent nodes: {parent.number_of_nodes()}, # of child nodes: {child.number_of_nodes()}, # of mapped pairs: {len(mappings)}\n\n")
            #print(mappings, f"# of parent nodes: {parent.graph['num_tensors']}, # of child nodes: {child.graph['num_tensors']}, # of mapped pairs: {len(mappings)}")
        if parent is not None:
            weights, num_of_matched, layer_mappings = self.warm_weights(parent, child, mappings)
            if num_of_matched > THRESHOLD * max(parent.graph['num_tensors'], child.graph['num_tensors']):
                parent_name = parent.graph['name']

                meta_data = {
                  "matching_score": best_score,
                  "parent_name": parent_name,
                  "parent_acc": parent.graph['accuracy'],
                  'num_of_matched': num_of_matched,
                  'parent_layers': parent.graph['num_tensors'],
                  'child_layers': child.graph['num_tensors'],
                  #'mappings': layer_mappings
                }
            logging.info(f"Querying model {child.graph['name']} completes with meta: {meta_data}")

        # remove the temporary onnx model
        os.remove(onnx_model_name)

        return weights, meta_data


    def map_for_onnx(self, child_onnx_path, blacklist=set(), model_name=None):
        """
        @ input are onnx models
        """
        child, child_onnx = self.load_model_meta(child_onnx_path)
        child.graph['model_id'] = str(self.current_mapping_id)

        # find the best mapping from the zoo
        if len(self.model_zoo) > self.mode_threshold:
            parent, mappings, best_score = self.query_best_mapping(child, blacklist, model_name)
        else:
            parent, mappings, best_score = self.get_best_mapping(child, blacklist, model_name)

        # overwrite the current model weights
        weights, num_of_matched = None, 0
        parent_name, meta_data = 'None', {}

        if mappings is not None and len(mappings) > 0:
        #    for p, s in mappings:
        #        if p != s:
        #           logging.info(f"Mismatch {parent.nodes[p]} to {child.nodes[s]}")
        #     assert (any([x[0]!=x[1] for x in mappings))
        #for n in range(parent.number_of_nodes()):
        #    logging.info(f"{n}, {parent.nodes[n]}, {parent.in_degree(n)}, {parent.out_degree(n)}")
            #logging.info(f"{[(n, parent.nodes[n]) for n in range(parent.number_of_nodes())]}")
            logging.info(f"{sorted(mappings)}\n# of parent nodes: {parent.number_of_nodes()}, # of child nodes: {child.number_of_nodes()}, # of mapped pairs: {len(mappings)}\n\n")

        if parent is not None:
            weights, num_of_matched, layer_mappings = self.warm_weights(parent, child, mappings)
            if num_of_matched > THRESHOLD * max(parent.graph['num_tensors'], child.graph['num_tensors']):
                parent_name = parent.graph['name']

                meta_data = {
                  "matching_score": best_score,
                  "parent_name": parent_name,
                  "parent_acc": parent.graph['accuracy'],
                  'num_of_matched': num_of_matched,
                  'parent_layers': parent.graph['num_tensors'],
                  'child_layers': child.graph['num_tensors'],
                  #'mappings': layer_mappings
                }
            logging.info(f"Querying model {child.graph['name']} completes with meta: {meta_data}")

            # update model value
            self.model_zoo[parent.graph['name']].increase_value(HIT_BENEFIT)

        return weights, meta_data


    def export_query_res(self, model_name, weights, meta_data):
        export_path = os.path.join(self.args.zoo_ans_path, model_name)
        with open(export_path, 'wb') as fout:
            pickle.dump(weights, fout)
            pickle.dump(meta_data, fout)

        ans = os.system(f"mv {export_path} {export_path.replace('.onnx', '.out')}")


    def schedule_job_order(self, models, threshold=0., timeout=120):
        """
        @ models: names of jobs needed for scheduling.
          - These models should have been added to zoo temporarily, and we can evict them once scheduling done
        @ threshold: disconnect the tree if score < threshold
        @ return: a list of jobs in scheduling order
        """

        self.add_to_zoo(models)

        zoo_models = list(self.model_zoo.keys())
        score_matrix = np.zeros([len(self.model_zoo), len(self.model_zoo)])

        start_time = time.time()
        graph_complete = False

        while time.time() - start_time < timeout and not graph_complete:
            graph_complete = True

            for model in zoo_models:
                for model_b in zoo_models:
                    if model not in self.distance or model_b not in self.distance[model]:
                        graph_complete = False
                        break
                if not graph_complete:
                    break

            if not graph_complete:
                logging.info(f"scheduler is working on matching scores ...")
                time.sleep(10)

        if not graph_complete:
            logging.warning(f"Better to have more time for matching scores. Current solution is suboptimal")

        for i, model_a in enumerate(zoo_models):
            for j, model_b in enumerate(zoo_models):
                if model_a in self.distance and model_b in self.distance[a]:
                    score_matrix[i][j] = self.distance[i][j]
                else:
                    score_matrix[i][j] = 0
            # avoid cycle
            score_matrix[i][i] = np.nan

        self.remove_from_zoo(models)

        from graphopt import GraphOperator

        scheduler = GraphOperator(threshold)

        temp_score_matrix = score_matrix.copy()
        score_matrix = np.transpose(score_matrix)
        job_order, mst_score = scheduler.max_spanning_tree(score_matrix)
        global_opt = scheduler.get_optimal(temp_score_matrix)
        trace_opt = scheduler.get_trace_optimal(temp_score_matrix)

        logging.info(f"Scheduling: global Optimal: {global_opt}, MST: {mst_score}, Trace Optimal: {trace_opt}")

        model_set = set(models)
        return [m for m in job_order if zoo_models[m] in model_set]


    def check_register_models(self):
        new_models = [x for x in os.listdir(self.args.zoo_register_path) if x.endswith('.onnx')]
        if len(new_models) > 0:
            for m in new_models:
                os.system(f'mv {os.path.join(self.args.zoo_register_path, m)} {self.args.zoo_path}')

            self.add_to_zoo([os.path.join(self.args.zoo_path, m) for m in new_models])
            logging.info(f"Added all pending models...")

    def check_pending_request(self):
        request_models = [x for x in os.listdir(self.args.zoo_query_path) if x.endswith('.onnx')]
        if len(request_models) > 0:
            for m in request_models:
                logging.info(f"Start to matching for model {m}")
                model_path = os.path.join(self.args.zoo_query_path, m)
                weights, meta_data = self.map_for_onnx(model_path, model_name=m)

                self.export_query_res(m, weights, meta_data)
                os.remove(model_path)
            logging.info(f"Served all queries, start gc collections...")

            gc.collect()


    def start(self):
        start_time = last_heartbeat = time.time()

        while True:

            # Register new models to zoo
            self.check_register_models()

            # Serve matching query
            self.check_pending_request()

            # Remove zoo
            #self.check_remove_request()

            time.sleep(2)

            if time.time() - last_heartbeat > 30:
                logging.info(f"ModelKeeper has been running {int(time.time() - start_time)} sec ...")
                last_heartbeat = time.time()


    def start_service(self):
        self.service_thread = threading.Thread(target=self.start)
        self.service_thread.setDaemon(True)
        self.service_thread.start()

        return

    def stop_service(self):
        try:
            if self.service_thread.isAlive():
                self.service_thread._stop()
        except Exception as e:
            # Python > 3.4 will throw errors
            pass

