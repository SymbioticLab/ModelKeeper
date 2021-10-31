import onnx
import networkx as nx
import time, sys, os
import functools, collections
import logging
from onnx import numpy_helper
import multiprocessing
import torch
from multiprocessing import Manager
import ctypes
import json
import concurrent, threading
import pickle
import gc
from itertools import repeat
import shutil
import random

sys.path.append('./modelkeeper')

from mappingopt import MappingOperator
# Libs for model clustering
from clustering import k_medoids

# Call C backend
clib_matcher = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), 'backend/bin/matcher.so'))
clib_matcher.get_matching_score.restype = ctypes.c_char_p

sys.setrecursionlimit(10000)
random.seed(1)
distance_lookup = None
SCORE_THRESHOLD = float('-inf')
THRESHOLD = 0.1 # more than X% layers can be transferred from the parent
MAX_MATCH_NODES=1000


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
            break
        elif '.bias' in _input:
            layer_name = _input
            break

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
    # return list(nx.topological_sort(graph))
    #visited = set()
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

            stack += temp_out #[edge[1] for edge in graph.out_edges(vertex) if edge[1] not in visited]

    # def dfs(node):
    #     visited.add(node)
    #     [dfs(edge[1]) for edge in graph.out_edges(node) if edge[1] not in visited]
    #     ret.append(node)

    [dfs_iterative(node) for node in graph.nodes() if graph.in_degree(node)==0]
    assert len(ret) == graph.number_of_nodes()
    # ret.reverse()
    return ret[:MAX_MATCH_NODES]

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
        self.parentPrevIndicesList = []

        self.match_res = None
        self.meta_data = {}

        self.num_of_nodes = len(self.parentidx_order)

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


def mapping_func(parent_opt, child, read_mapping=False, return_child_name=False):
    parent_name = parent_opt.parent.graph['name']

    try:
        score = parent_opt.get_matching_score(child=child, read_mapping=read_mapping)
    except Exception as e:
        logging.error(f"Mapping {parent_name} to {child.graph['name']} failed as {e}")
        score = float('-inf')

    if return_child_name:
        return parent_name, score, child.graph['name']
    return parent_name, score


class ModelKeeper(object):

    def __init__(self, args):
        self.args = args

        self.model_zoo = collections.OrderedDict()

        self.current_mapping_id = 0
        self.zoo_model_id = 0
        self.mode_threshold = 500 # enable clustering if len(model_zoo) > T

        self.model_clusters = []
        self.query_model = None

        # Storage of the model pair distance: {parent_model: {child_model}}
        self.distance = collections.defaultdict(dict)

        # Lock for offline clustering
        self.zoo_lock = threading.Lock()

        # Blacklist opts in matching
        self.skip_opts = {'Constant'}

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

        is_update = False
        for model_path in model_paths:
            if model_path in self.model_zoo:
                logging.warning(f"{model_path} is already in the zoo")
            else:
                try:
                    model_graph, model_weight = self.load_model_meta(model_path)
                    model_graph.graph['model_id'] = str(self.zoo_model_id)

                    with self.zoo_lock:
                        self.model_zoo[model_path] = MatchingOperator(parent=model_graph)

                    self.zoo_model_id += 1
                    is_update = True

                    logging.info(f"Added {model_path} to zoo ...")
                except Exception as e:
                    logging.info(f"Error: {e} for {model_path}")

        # Update model zoo
        if is_update and self.mode_threshold < len(self.model_zoo):
            self.update_model_clusters()


    def remove_from_zoo(self, model_paths):

        if not isinstance(model_paths, list):
            model_paths = [model_paths]

        for model_path in model_paths:
            if model_path in self.model_zoo:
                with self.zoo_lock:
                    del self.model_zoo[model_path]
            else:
                logging.warning(f"Fail to remove {model_path} from zoo, as it does not exist")

        # Update model zoo asyn
        if self.mode_threshold < len(self.model_zoo):
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

        with self.zoo_lock:
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


    def load_model_meta(self, meta_file='sample__accuracy.onnx'):
        """
        @ meta_file: input files are onnx. return the weight meta graph of this model
        """

        start_time = time.time()

        # meta file is rather small
        onnx_model = onnx.load(meta_file)
        model_graph = onnx_model.graph
        if '@' in meta_file:
            accuracy = float(meta_file.split('@')[-1].split('.onnx')[0])
        else:
            accuracy = -1

        # record the shape of each weighted nodes
        node_shapes, num_of_trainable_tensors = get_tensor_shapes(model_graph)

        # construct the computation graph and align their attribution
        nodes = [n for n in onnx_model.graph.node if n.op_type not in self.skip_opts]
        graph = nx.DiGraph(name=meta_file, num_tensors=num_of_trainable_tensors, accuracy=accuracy,
                            num_nodes=len(nodes))

        node_ids = dict()
        edge_source = collections.defaultdict(list)

        opt_dir = collections.defaultdict(int)

        for idx, node in enumerate(nodes):
            input_nodes, trainable_weights = split_inputs(node.input)
            opt_dir[node.op_type] += 1

            #logging.info(node.input, trainable_weights)
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


    def get_mappings(self, parent_path):
        mapping_res, score = self.model_zoo[parent_path].get_mappings()
        return (self.model_zoo[parent_path].parent, mapping_res, score)


    def query_scores(self, parents, child, threads=40):
        pool = multiprocessing.Pool(processes=threads)
        results = [pool.apply_async(mapping_func, (self.model_zoo[parent], child)) for parent in parents]

        pool.close()
        pool.join()

        scores = [res.get() for res in results]

        for (p, s) in scores:
            self.distance[p][child.graph['name']] = 1. - s

        return scores


    def mapping_func(self, model, child=None):
        query_model = self.query_model if child is None else child
        return mapping_func(model, query_model)


    def query_best_mapping(self, child, blacklist=set(),
                            model_name=None, return_weight=True,
                            score_threshold=0.95, timeout=60):

        start_time = time.time()
        self.query_model = child

        # 1. Pick top-k clusters
        medoids = [medoid.kernel for medoid in self.model_clusters]
        medoid_dist = []

        if len(medoids) > 0:
            medoid_dist = self.query_scores(medoids, child, self.args.num_of_processes)
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

        with concurrent.futures.ProcessPoolExecutor() as executor:
            #for model, score in executor.map(self.mapping_func, search_models, timeout=timeout):
            for model, score in executor.map(mapping_func, search_models, repeat(child), timeout=timeout):
                if score > best_score:
                    parent_path, best_score = model, score

                self.distance[model][child.graph['name']] = 1.0-score

                if best_score >= score_threshold:
                    executor.shutdown(wait=False)
                    break

        if parent_path is not None and return_weight:
            mapping_func(self.model_zoo[parent_path], child, read_mapping=True)
            parent, mappings, _ = self.get_mappings(parent_path)

        if parent is not None:
            logging.info("{} find best mappings {} (score: {}) takes {:.2f} sec\n\n".format(
                        child.graph['name'], parent.graph['name'], round(best_score,4), time.time() - start_time))
        else:
            logging.info("{} does not find best mapping, takes {:.2f} sec\n\n".format(
                        child.graph['name'], time.time() - start_time))

        return parent, mappings, best_score


    def get_best_mapping(self, child, blacklist=set(), model_name=None, return_weight=True):
        """
            Enumerate all possible model pairs. Not as efficient as the clustering one.
        """
        start_time = time.time()
        self.query_model = child

        parent_models = [model for model in self.model_zoo.keys() if model not in blacklist]
        results = self.query_scores(parent_models, child, self.args.num_of_processes)

        parent_path = mappings = parent = None
        best_score = SCORE_THRESHOLD

        for (p, s) in results:
            #logging.info(f"For mapping pair ({model_name}, {p.graph['name']}) score is {s}")
            logging.info(f"For mapping pair ({model_name}, {p}) score is {s}")
            if s > best_score:
                parent_path, best_score = p, s

        if parent_path is not None and return_weight:
            mapping_func(self.model_zoo[parent_path], child, read_mapping=True)
            parent, mappings, _ = self.get_mappings(parent_path)

        if parent is not None:
            logging.info("{} find best mappings {} (score: {}) takes {:.2f} sec\n\n".format(
                        child.graph['name'], parent.graph['name'], round(best_score,4), time.time() - start_time))
        else:
            logging.info("{} does not find best mapping, takes {:.2f} sec\n\n".format(
                        child.graph['name'], time.time() - start_time))

        return parent, mappings, best_score


    def warm_weights(self, parent, child, mappings):

        mapper = MappingOperator(parent, child, mappings)
        mapper.cascading_mapping()
        mapper.pad_mapping()
        weights, num_of_matched = mapper.get_mapping_weights()

        logging.info("Mapped {} layers to the child model ({} layers), parent {} layers".format(num_of_matched, child.graph['num_tensors'],
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

        if parent is not None and len(mappings) > THRESHOLD * parent.graph['num_nodes']:
            weights, num_of_matched = self.warm_weights(parent, child, mappings)
            parent_name = parent.graph['name']

            meta_data = {
              "matching_score": best_score,
              "parent_name": parent_name,
              "parent_acc": parent.graph['accuracy']
            }

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

        if parent is not None and len(mappings) > THRESHOLD * parent.graph['num_nodes']:
            weights, num_of_matched = self.warm_weights(parent, child, mappings)
            parent_name = parent.graph['name']

            meta_data = {
              "matching_score": best_score,
              "parent_name": parent_name,
              "parent_acc": parent.graph['accuracy'],
              'num_of_matched': num_of_matched,
              'parent_layers': parent.graph['num_tensors']
            }

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


    def check_pending_request(self):
        request_models = [x for x in os.listdir(self.args.zoo_query_path) if x.endswith('.onnx')]
        if len(request_models) > 0:
            for m in request_models:
                logging.info(f"Start to matching for model {m}")
                model_path = os.path.join(self.args.zoo_query_path, m)
                weights, meta_data = self.map_for_onnx(model_path, model_name=m)

                self.export_query_res(m, weights, meta_data)
                os.remove(model_path)


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
                gc.collect()


    def start_service(self):
        self.service_thread = threading.Thread(target=self.start)
        self.service_thread.setDaemon(True)
        self.service_thread.start()

    def stop_service(self):
        try:
            if self.service_thread.isAlive():
                self.service_thread._stop()
        except Exception as e:
            # Python > 3.4 will throw errors
            pass


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

    logging.info(opt.alignmentStrings()[0])
    logging.info("\n\n")
    logging.info(opt.alignmentStrings()[1])
    logging.info("\n\n")
    logging.info(opt.graphStrings()[0])
    logging.info("\n\n")
    logging.info(opt.graphStrings()[1])
    return (parent, mappings, score)

def test_fake():
    parent, child = faked_graph(), faked_graph2()

    mapper = MatchingOperator(parent=parent)
    logging.info(mapper.get_mappings(child))


def test():
    # import argparse

    # start_time = time.time()
    # zoo_path = '/mnt/zoo/tests/'

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--zoo_path', type=str, default=zoo_path)
    # parser.add_argument('--num_of_processes', type=int, default=30)
    # parser.add_argument('--neigh_threshold', type=float, default=0.05)

    # args = parser.parse_args()
    from config import modelkeeper_config
    zoo_path = '/users/fanlai/experiment/exp_logs/keeper/model_zoo'
    modelkeeper_config.zoo_path = zoo_path

    mapper = ModelKeeper(modelkeeper_config)

    #child_onnx_path = '/mnt/zoo/tests/vgg11.onnx'
    models = os.listdir(zoo_path)

    for model in models:
        child_onnx_path = os.path.join(zoo_path, model)
        weights, meta_data = mapper.map_for_onnx(child_onnx_path, blacklist=set([child_onnx_path]))

        logging.info("\n\nMatching {}, results: {}\n".format(child_onnx_path, meta_data))

    # time.sleep(40)

#test()
#test_fake()


