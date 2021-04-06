
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

sys.setrecursionlimit(10000)
#logging.basicConfig(filename='logging', level=logging.INFO)

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
    __matchscore = 1
    __mismatchscore = -1
    __gap = -2

    def __init__(self, parent, globalAlign=True,
                 matchscore=__matchscore, mismatchscore=__mismatchscore,
                 gapscore=__gap):

        self._mismatchscore = mismatchscore
        self._matchscore = matchscore
        self._gap = gapscore
        self.parent       = parent
        self.matchidxs  = None
        self.parentidxs    = None
        self.globalAlign = globalAlign
        self.child = None
        self.childidx_order = None

        # parent
        self.nodeIDtoIndex = {}
        self.nodeIndexToID = {-1: None}

        # child
        self.childNodeIDtoIndex = {}

        self.parentidx_order = topological_sorting(self.parent)
        self.parentPrevIndicesList = []

        self.child_cache = {}
        self.parent_cache = {}
        self.match_res = None

    def align_child(self, child):
        start_time = time.time()

        # reset all parameters
        self.child = child
        self.matchidxs = self.parentidxs = None
        self.parentPrevIndicesList = []
        self.child_cache = {}
        self.childNodeIDtoIndex = {}
        self.moveTypes = {}
        self.match_res = None

        self.childidx_order = topological_sorting(self.child)
        self.child_bases = {cidx:self.child.nodes[cidx]['attr'] for j, cidx in enumerate(self.childidx_order)}

        """Align node to parent, following same approach as smith waterman example"""
        scores, backStrIdx, backGrphIdx = self.initializeDynamicProgrammingData()

        #matches = self.alignStringToGraphFast()
        self.match_res = self.alignChildToParent(scores, backStrIdx, backGrphIdx)

        #print("Match {} takes {:.2f} sec".format(self.parent.graph['name'], time.time() - start_time))
        return self.match_res[0][-1][-1] # score of end-to-end matching

    def alignmentStrings(self):
        return ("\t".join([self.parent.nodes[j]['attr']['name'] if j is not None else "-" for j in self.parentidxs]), 
                "\t".join([self.child.nodes[i]['attr']['name'] if i is not None else "-" for i in self.matchidxs]),
                self.match_score)

    def graphStrings(self):
        return ("\t".join([self.parent.nodes[j]['attr']['name'] if j is not None else "-" for j in self.parentidx_order]),
                "\t".join([self.child.nodes[i]['attr']['name'] if i is not None else "-" for i in self.childidx_order])
                )

    def get_mappings(self, child):
        #if self.match_res is None:
        score = self.align_child(child=child)
        matches = self.backtrack(*self.match_res)
        self.matchidxs, self.parentidxs, self.match_score = matches

        mapping_res = []
        parent_set = set()
        child_set = set()

        for i in range(len(self.parentidxs)):
            if self.parentidxs[i] is not None and self.matchidxs[i] is not None:
                if self.parentidxs[i] not in parent_set and self.matchidxs[i] not in child_set:
                    mapping_res.append((self.parentidxs[i], self.matchidxs[i]))

                    parent_set.add(self.parentidxs[i])
                    child_set.add(self.matchidxs[i])

        return mapping_res, self.match_score

    def matchscore(self, parent_opt, child_opt):
        if parent_opt['op_type'] != child_opt['op_type']:
            return self._mismatchscore
        else:
            # diff number of parameters
            num_param_p = self.get_parent_parameters(parent_opt['name'], parent_opt['dims'])
            #num_param_c = self.get_child_parameters(child_opt['name'], child_opt['dims'])
            num_inherited = 1.
            for i, j in zip(parent_opt['dims'], child_opt['dims']):
                num_inherited *= min(i, j)

            match_score = float(num_inherited)/max(num_param_p, 1e-4) * self._matchscore

            #print(parent_opt['name'], child_opt['name'], parent_opt['dims'], child_opt['dims'], match_score)
            return match_score #if match_score > .25 else self._mismatchscore


    def get_child_parameters(self, name, dims):
        if name not in self.child_cache:
            self.child_cache[name] = numpy.prod(dims)
        return self.child_cache[name]

    def get_parent_parameters(self, name, dims):
        if name not in self.parent_cache:
            self.parent_cache[name] = numpy.prod(dims)
        return self.parent_cache[name]

    def matchscoreVec(self, parent_opt, child_opts):
        res = []
        parent_opt_type = parent_opt['op_type']
        parent_num_params = numpy.prod(parent_opt['dims'])

        for child_opt in child_opts:
            if parent_opt_type != child_opt['op_type']:
                res.append(self._mismatchscore)
            else:
                # diff number of parameters
                child_num_params = self.get_child_parameters(child_opt['name'], child_opt['dims'])
                match_score = (1.-abs(parent_num_params-child_num_params)/max(parent_num_params, child_num_params, 1e-4)) * self._matchscore
                res.append(match_score)

        return res


    def alignStringToGraphFast(self):
        """Align string to graph - using numpy to vectorize across the string
        at each iteration."""

        scores, backStrIdx, backGrphIdx = self.initializeDynamicProgrammingData()

        l2 = len(self.child.nodes)
        inserted = numpy.zeros((l2), dtype=numpy.bool)

        sbases = [self.child.nodes[cidx]['attr'] for j, cidx in enumerate(self.childidx_order)]
        seqvec = numpy.array(list(sbases))

        # having the inner loop as a function improves performance
        # can use Cython, etc on this for significant further improvements
        # can't vectorize this since there's a loop-carried dependency
        #  along the string
        def insertions(i, l2, scores, inserted):
            inserted[:] = False
            for j in range(l2):
                insscore = scores[i+1, j] + self._gap
                if insscore >= scores[i+1, j+1]:
                    scores[i+1, j+1] = insscore
                    inserted[j] = True

        # Dynamic Programming
        for i, node in enumerate(self.parentidx_order):
            gbase = self.parent.nodes[node]['attr']
            predecessors = self.parentPrevIndicesList[i]

            # calculate all best deletions, matches in one go over all
            # predecessors.

            # First calculate for the first predecessor, over all string posns:
            deletescore = scores[predecessors[0]+1, 1:] + self._gap
            bestdelete = numpy.zeros((l2), dtype=numpy.int)+predecessors[0]+1

            matchpoints = self.matchscoreVec(gbase, seqvec)
            matchscore = scores[predecessors[0]+1, 0:-1] + matchpoints
            bestmatch = numpy.zeros((l2), dtype=numpy.int)+predecessors[0]+1

            # then, the remaining
            for predecessor in predecessors[1:]:
                newdeletescore = scores[predecessor+1, 1:] + self._gap
                bestdelete     = numpy.where(newdeletescore > deletescore, predecessor+1, bestdelete)
                deletescore    = numpy.maximum(newdeletescore, deletescore)

                gbase = self.parent.nodes[predecessor]['attr']
                matchpoints = self.matchscoreVec(gbase, seqvec)
                newmatchscore = scores[predecessor+1, 0:-1] + matchpoints
                bestmatch     = numpy.where(newmatchscore > matchscore, predecessor+1, bestmatch)
                matchscore    = numpy.maximum(newmatchscore, matchscore)

            # choose best options available of match, delete
            deleted       = deletescore >= matchscore
            backGrphIdx[i+1, 1:] = numpy.where(deleted, bestdelete, bestmatch)
            backStrIdx [i+1, 1:] = numpy.where(deleted, numpy.arange(1, l2+1), numpy.arange(0, l2))
            scores[i+1, 1:] = numpy.where(deleted, deletescore, matchscore)

            # insertions: updated in place, don't depend on predecessors
            insertions(i, l2, scores, inserted)
            backGrphIdx[i+1, 1:] = numpy.where(inserted, i+1, backGrphIdx[i+1, 1:])
            backStrIdx[i+1, 1:] = numpy.where(inserted, numpy.arange(l2), backStrIdx[i+1, 1:])


        return self.backtrack(scores, backStrIdx, backGrphIdx)

    def alignChildToParent(self, scores, backStrIdx, backGrphIdx):

        # Dynamic Programming
        for i, pidx in enumerate(self.parentidx_order):
            pbase = self.parent.nodes[pidx]['attr']

            for j, cidx in enumerate(self.childidx_order):
                sbase = self.child_bases[cidx]
                match_score = self.matchscore(pbase, sbase)
                best_candidates = (float('-inf'), )

                for cp in self.childPrevIndicesList[cidx]:
                    # index of child parent
                    cprev = cp+1
                    best_candidates = max(best_candidates, (scores[i+1, cprev] + self._gap, i+1, cprev))#, "INS")) # skip a parent node
                    
                    # add all candidates to a list, pick the best insert to the child
                    for predIndex in self.parentPrevIndicesList[i]:
                        best_candidates = max(best_candidates, 
                                        (scores[predIndex+1, cprev] + match_score, predIndex+1, cprev),#, "MATCH"),
                                        (scores[predIndex+1, cprev+1] + self._gap, predIndex+1, cprev+1))#, "DEL")) # skip a child node

                scores[i+1, j+1], backGrphIdx[i+1, j+1], backStrIdx[i+1, j+1] = best_candidates

        return scores, backStrIdx, backGrphIdx 


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

    def initializeDynamicProgrammingData(self):
        """Initalize the dynamic programming tables:
            @ni: re-index graph nodes
            - set up scores array
            - set up backtracking array
            - create index to Node ID table and vice versa
        """
        l1 = self.parent.number_of_nodes()
        l2 = self.child.number_of_nodes()

        self.nodeIDtoIndex = {}
        # generate a dict of (nodeID) -> (index into nodelist (and thus matrix))
        for (index, nidx) in enumerate(self.parentidx_order):
            self.nodeIDtoIndex[nidx] = index

        for (index, nidx) in enumerate(self.childidx_order):
            self.childNodeIDtoIndex[nidx] = index

        # initiate prevs for parent and child graph
        self.parentPrevIndicesList = []
        for (index, nidx) in enumerate(self.parentidx_order):
            self.parentPrevIndicesList.append(self.parentPrevIndices(nidx))

        self.childPrevIndicesList = {}
        for (index, nidx) in enumerate(self.childidx_order):
            self.childPrevIndicesList[nidx] = self.childPrevIndices(nidx)

        # Dynamic Programming data structures; scores matrix and backtracking
        # matrix
        scores = numpy.zeros((l1+1, l2+1), dtype=numpy.float)

        # initialize insertion score
        # if global align, penalty for starting at head != 0
        if self.globalAlign:
            scores[0, :] = numpy.arange(l2+1)*self._gap
            scores[:, 0] = numpy.arange(l1+1)*self._gap

            for (index, nidx) in enumerate(self.parentidx_order):
                prevIdxs = self.parentPrevIndicesList[index]

                best = float('-inf')
                for prevIdx in prevIdxs:
                    best = max(best, scores[prevIdx+1, 0])
                scores[index+1, 0] = best + self._gap

            for (index, nidx) in enumerate(self.childidx_order):
                prevIdxs = self.childPrevIndicesList[nidx]

                best = float('-inf')
                for prevIdx in prevIdxs:
                    best = max(best, scores[0, prevIdx+1])
                scores[0, index+1] = best + self._gap

        # backtracking matrices
        backStrIdx = numpy.zeros((l1+1, l2+1), dtype=numpy.int)
        backGrphIdx = numpy.zeros((l1+1, l2+1), dtype=numpy.int)

        return scores, backStrIdx, backGrphIdx

    def backtrack(self, scores, backStrIdx, backGrphIdx):
        """Backtrack through the scores and backtrack arrays.
           Return a list of child indices and node IDs (not indices, which
           depend on ordering)."""
        besti, bestj = scores.shape
        besti -= 1
        bestj -= 1

        bestscore = scores[besti, bestj]
        matches, strindexes = [], []
        que = [(besti, bestj)]
        que_set = set([(besti, bestj)])

        # print(scores)

        # print(backGrphIdx[6, :])
        # print(backStrIdx[6, :])

        while len(que) != 0:
            besti, bestj = que.pop()
            curstridx, curnodeidx = self.childidx_order[bestj-1], self.parentidx_order[besti-1]

            nextis, nextjs = [backGrphIdx[besti, bestj]], [backStrIdx[besti, bestj]]
            # multi-branch for child (j), then we need to recover other branches
            for childPrev in self.childPrevIndicesList[curstridx]:
                # branch may conflict
                if childPrev + 2 != bestj:
                    # recover how to reach the current node
                    nextis.append(backGrphIdx[besti, childPrev+1])
                    nextjs.append(backStrIdx[besti, childPrev+1])

                #print()

            #print([self.parentidx_order[x] for x in nextis], [self.childidx_order[x] for x in nextjs], curstridx, [self.childidx_order[x] for x in self.childPrevIndicesList[curstridx]])
            #nextis, nextjs = [backGrphIdx[besti, bestj]], [backStrIdx[besti, bestj]]

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

        return strindexes, matches, bestscore

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

        for model_path in model_paths:
            self.add_to_zoo(model_path)
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

        #print('\nLoad {} takes {} sec'.format(meta_file, time.time() - start_time))

        return graph, onnx_model


    def add_to_zoo(self, model_path):
        model_graph, model_weight = self.load_model_meta(model_path)
        self.model_zoo[model_path] = MatchingOperator(parent=model_graph)


    def remove_from_zoo(self, model_path):
        if model_path in self.model_zoo:
            del self.model_zoo[model_path]
        else:
            logging.warning(f"Fail to remove {model_path} from zoo, as it does not exist")


    def mapping_func(self, parent_path, child):
        mappings, score = self.model_zoo[parent_path].get_mappings(child=child)

        return (self.model_zoo[parent_path].parent, mappings, score)


    def get_best_mapping(self, child):

        start_time = time.time()

        pool = multiprocessing.Pool(processes=self.args.num_of_processes)
        results = []

        for model_path in self.model_zoo.keys(): 
            results.append(pool.apply_async(self.mapping_func, (model_path, child)))
        pool.close()
        pool.join()

        parent, mappings, best_score = None, [], float('-inf')

        for res in results:
            (p, m, s) = res.get()
            if s > best_score:
                parent, mappings, best_score = p, m, s

        if parent is not None:
            print("Find best mappings {} (score: {}) takes {:.2f} sec\n\n".format(
                                parent.graph['name'], best_score, time.time() - start_time))
        return parent, mappings, best_score


    def warm_weights(self, parent, child, mappings):

        mapper = MappingOperator(parent, child, mappings)
        mapper.cascading_mapping()
        mapper.pad_mapping()
        weights, num_of_matched = mapper.get_mapping_weights()

        # get_warmed child model
        # for name, p in child_model.named_parameters():
        #     p.data = (torch.from_numpy(weights[name])).data

        print("Mapped {} layers to the child model ({} layers)".format(num_of_matched, child.graph['num_tensors']))

        return weights, num_of_matched

    def map_for_model(self, child_model, dummy_input, hidden = None):

        self.current_mapping_id += 1

        # dump the model into onnx format
        onnx_model_name = os.path.join(self.args.exe_path, str(self.current_mapping_id)+".onnx")
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

        # find the best mapping from the zoo
        parent, mappings, best_score = self.get_best_mapping(child)

        # overwrite the current model weights
        weights, num_of_matched = None, 0
        if parent is not None:
            weights, num_of_matched = self.warm_weights(parent, child, mappings)

        # remove the temporary onnx model
        os.remove(onnx_model_name)

        return weights, num_of_matched

def faked_graph():
    graph = nx.DiGraph(name='faked')
    attr1={'dims': [64, 32, 3, 3], 'op_type': 'cov1',}

    graph.add_node(0, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'0'})

    graph.add_node(1, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'1'})
    graph.add_node(2, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'2'})

    graph.add_node(5, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'5'})
    graph.add_node(3, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'3'})

    graph.add_node(4, attr={'dims': [5, 32, 3, 3], 'op_type': 'cov1', 'name':'4'})

    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(2, 4)
    graph.add_edge(0, 5)
    graph.add_edge(5, 3)
    graph.add_edge(3, 4)

    return graph

def faked_graph2():
    graph = nx.DiGraph(name='faked')
    attr1={'dims': [64, 32, 3, 3], 'op_type': 'cov1',}

    graph.add_node(0, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'0'})

    graph.add_node(1, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'1'})
    graph.add_node(2, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'2'})

    graph.add_node(5, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'5'})
    graph.add_node(3, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'3'})

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


def main():
    start_time = time.time()
    num_of_processes = 16
    # parent = faked_graph()
    # child = faked_graph2()#parent.copy()

    # parent, mappings, best_score = mapping_faked(parent, child)
    # print("Find best mappings {} (score: {}) takes {:.2f} sec\n\n".format(parent.graph['name'], best_score, time.time() - start_time))


    #zoo_path = './temp_zoo'
    #zoo_path = './zoo/resnet18.onnx'
    zoo_path = './zoo/densenet201.onnx'
    model_zoo = get_model_zoo(zoo_path)

    # create multiple process to handle model zoos
    #child, child_onnx = load_model_meta('./zoo/resnet152.onnx')
    child, child_onnx = load_model_meta('./zoo/densenet201.onnx')

    results = []
    pool = multiprocessing.Pool(processes=num_of_processes)

    for model in model_zoo: 
        results.append(pool.apply_async(mapping_func, (model, child)))
    pool.close()
    pool.join()

    best_score = float('-inf')
    parent = mappings = None

    for res in results:
        (p, m, s) = res.get()
        if s > best_score:
            parent, mappings, best_score = p, m, s

    print("Find best mappings {} (score: {}) takes {:.2f} sec\n\n".format(parent.graph['name'], best_score, time.time() - start_time))

    #print(mappings)
    mapper = MappingOperator(parent, child, mappings)
    mapper.cascading_mapping()
    mapper.pad_mapping()
    weights, num_of_matched = mapper.get_mapping_weights()

    # record the shape of each weighted nodes
    for idx, key in enumerate(weights.keys()):
        child_onnx.graph.initializer[idx].CopyFrom(numpy_helper.from_array(weights[key]))

    print("\n\n{} layers in total, matched {} layers".format(child.graph['num_tensors'], num_of_matched))
    #onnx.save(child_onnx, child.graph['name']+'_new.onnx')

    print("\n\n")
    print("Match takes {:.2f} sec".format(time.time() - start_time))


