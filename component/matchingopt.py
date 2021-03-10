
import onnx
import numpy
import networkx as nx
import time, sys, os
import functools, collections
from mappingopt import MappingOperator
import logging
from onnx import numpy_helper
import multiprocessing
#from numba import jit

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

skip_opts = {'Constant'}

def load_model_meta(meta_file='sample.onnx'):
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
            'name': node.name if node.name else str(node.op_type)+str(opt_dir[node.op_type]),
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

def clip(var):
    if var < -1.: return -1.
    if var > 1.: return 1.
    return var

def topological_sorting(graph):
    """DFS based topological sort to maximize length of each chain"""
    visited = set()
    ret = []

    def dfs(node):
        visited.add(node)
        [dfs(edge[1]) for edge in graph.out_edges(node) if edge[1] not in visited]
        ret.append(node)

    [dfs(node) for node in graph.nodes() if graph.in_degree(node)==0]
    ret.reverse()
    return ret

def contract_graph(graph):
    """Contract all branches into new subgraphs"""
    branch_source = set([node for node in graph.nodes() if graph.out_degree(node)>1])
    branch_sink = set([node for node in graph.nodes() if graph.in_degree(node)>1])

    blockwise_graph = networkx.DiGraph()
    block_idx = 1e5 # avoid conflicting with current node idx

    def dfs(cur_graph, node, parents=None):
        nonlocal block_idx

        temp_graph = None
        cur_graph.add_node(node, attr=graph.nodes[node]['attr'])

        # not the beginning or end of a new subgraph
        if node not in branch_source and node not in branch_sink:
            temp_graph = cur_graph

        elif node in branch_source:
            # create a new subgraph, padding a super node to current graph
            # insert a block node
            sub_graph = networkx.DiGraph()
            cur_graph.add_node(block_idx, attr={'subgraph':subgraph})
            temp_graph = sub_graph
        else:
            # end of a subgraph
            pass
        for source, target in graph.out_edges(node):
            pass

        return graph

    [dfs(blockwise_graph, node) for node in graph.nodes() if graph.in_degree(node)==0]

    return blockwise_graph

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

        self.nodeIDtoIndex = {}
        self.nodeIndexToID = {-1: None}

        self.parentidx_order = topological_sorting(self.parent)
        self.parentPrevIndicesList = []

        self.child_cache = {}

    def align_child(self, child):
        start_time = time.time()
        self.child = child
        self.matchidxs = self.parentidxs = None
        self.parentPrevIndicesList = []
        self.child_cache = {}

        self.childidx_order = topological_sorting(self.child)
        matches = self.alignStringToGraphFast()
        #matches = self.alignChildToParent()
        self.matchidxs, self.parentidxs, self.match_score = matches
        print("Match {} takes {:.2f} sec".format(self.parent.graph['name'], time.time() - start_time))

    def alignmentStrings(self):
        return ("\t".join([self.parent.nodes[j]['attr']['name'] if j is not None else "-" for j in self.parentidxs]), 
                "\t".join([self.child.nodes[i]['attr']['name'] if i is not None else "-" for i in self.matchidxs]),
                self.match_score)

    def graphStrings(self):
        return ("\t".join([self.parent.nodes[j]['attr']['name'] if j is not None else "-" for j in self.parentidx_order]),
                "\t".join([self.child.nodes[i]['attr']['name'] if i is not None else "-" for i in self.childidx_order])
                )

    def get_mappings(self, child):
        self.align_child(child)
        return [(self.parentidxs[i], self.matchidxs[i]) for i in range(len(self.parentidxs)) \
                if self.parentidxs[i] is not None and self.matchidxs[i] is not None], self.match_score

    def matchscore(self, parent_opt, child_opt):
        if parent_opt['op_type'] != child_opt['op_type']:
            return self._mismatchscore
        else:
            # diff number of parameters
            num_param_p = numpy.prod(parent_opt['dims'])
            num_param_c = numpy.prod(child_opt['dims'])

            match_score = (1.-abs(num_param_p-num_param_c)/max(num_param_p, num_param_c, 1e-4)) * self._matchscore

            return match_score #if match_score > .25 else self._mismatchscore

    def get_child_parameters(self, name, dims):
        if name not in self.child_cache:
            self.child_cache[name] = numpy.prod(dims)
        return self.child_cache[name]

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

    def get_max(self, candidates):
        ans = 0

        for i in range(1, len(candidates)):
            if (candidates[ans][0], candidates[ans][-1]) <= (candidates[i][0], candidates[i][-1]):
                ans = i
        return candidates[ans]

    def alignStringToGraphFast(self):
        """Align string to graph - using numpy to vectorize across the string
        at each iteration."""

        scores, backStrIdx, backGrphIdx = self.initializeDynamicProgrammingData(self.parentidx_order)

        l2 = len(self.child.nodes)
        inserted = numpy.zeros((l2), dtype=numpy.bool)

        sbases = [self.child.nodes[cidx]['attr'] for j, cidx in enumerate(self.childidx_order)]
        seqvec = numpy.array(list(sbases))

        # having the inner loop as a function improves performance
        # can use Cython, etc on this for significant further improvements
        # can't vectorize this since there's a loop-carried dependency
        #  along the string
        #@jit
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

            # if we're doing local alignment, don't let bad global alignment
            # drag us negative
            # if not self.globalAlign:
            #     backGrphIdx[i+1, :] = numpy.where(scores[i+1, :] > 0, backGrphIdx[i+1, :], -1)
            #     backStrIdx [i+1, :] = numpy.where(scores[i+1, :] > 0, backStrIdx[i+1, :], -1)
            #     scores[i+1, :]      = numpy.maximum(scores[i+1, :], 0)

        return self.backtrack(scores, backStrIdx, backGrphIdx, self.parentidx_order)

    #@jit
    def alignChildToParent(self):
        """Align node to parent, following same approach as smith waterman
        example"""
        scores, backStrIdx, backGrphIdx = self.initializeDynamicProgrammingData(self.parentidx_order)

        # make it a list for speedup
        sbases = [self.child.nodes[cidx]['attr'] for j, cidx in enumerate(self.childidx_order)]
        
        # Dynamic Programming
        for i, pidx in enumerate(self.parentidx_order):
            pbase = self.parent.nodes[pidx]['attr']

            for j, cidx in enumerate(self.childidx_order):
                sbase = sbases[j]
                candidates = [(scores[i+1, j] + self._gap, i+1, j, "INS")] # skip a parent node
                # add all candidates to a list, pick the best
                # insert to the child
                for predIndex in self.parentPrevIndicesList[i]:
                    candidates += [(scores[predIndex+1, j] + self.matchscore(pbase, sbase), predIndex+1, j, "MATCH")]
                    candidates += [(scores[predIndex+1, j+1] + self._gap, predIndex+1, j+1, "DEL")] # skip a child node

                scores[i+1, j+1], backGrphIdx[i+1, j+1], backStrIdx[i+1, j+1], movetype = max(candidates)

                # if not self.globalAlign and scores[i+1, j+1] < 0:
                #     scores[i+1, j+1] = 0.
                #     backGrphIdx[i+1, j+1] = -1
                #     backStrIdx[i+1, j+1] = -1

        return self.backtrack(scores, backStrIdx, backGrphIdx, self.parentidx_order)

    # networkx is too slow in accessing edges
    def parentPrevIndices(self, node):
        """Return a list of the previous dynamic programming table indices
           corresponding to predecessors of the current node."""
        prev = []
        for edge in self.parent.in_edges(node):
            prev.append(self.nodeIDtoIndex[edge[0]])

        #prev.sort(key=lambda k:-self.parentidx_order.index(k))

        # if no predecessors, point to just before the parent
        if len(prev) == 0:
            prev = [-1]
        return prev

    def initializeDynamicProgrammingData(self, ni):
        """Initalize the dynamic programming tables:
            @ni: re-index graph nodes
            - set up scores array
            - set up backtracking array
            - create index to Node ID table and vice versa
        """
        l1 = self.parent.number_of_nodes()
        l2 = self.child.number_of_nodes()

        self.nodeIDtoIndex = {}
        self.nodeIndexToID = {-1: None}
        # generate a dict of (nodeID) -> (index into nodelist (and thus matrix))
        for (index, nidx) in enumerate(ni):
            self.nodeIDtoIndex[nidx] = index
            self.nodeIndexToID[index] = nidx

        # Dynamic Programming data structures; scores matrix and backtracking
        # matrix
        scores = numpy.zeros((l1+1, l2+1), dtype=numpy.float)

        # initialize insertion score
        # if global align, penalty for starting at head != 0
        if self.globalAlign:
            scores[0, :] = numpy.arange(l2+1)*self._gap
            scores[:, 0] = numpy.arange(l1+1)*self._gap

            for (index, nidx) in enumerate(ni):
                prevIdxs = self.parentPrevIndices(nidx)
                self.parentPrevIndicesList.append(prevIdxs)

                best = float('-inf')
                for prevIdx in prevIdxs:
                    best = max(best, scores[prevIdx+1, 0])
                scores[index+1, 0] = best + self._gap

        # backtracking matrices
        backStrIdx = numpy.zeros((l1+1, l2+1), dtype=numpy.int)
        backGrphIdx = numpy.zeros((l1+1, l2+1), dtype=numpy.int)

        return scores, backStrIdx, backGrphIdx

    def backtrack(self, scores, backStrIdx, backGrphIdx, ni):
        """Backtrack through the scores and backtrack arrays.
           Return a list of child indices and node IDs (not indices, which
           depend on ordering)."""
        besti, bestj = scores.shape
        besti -= 1
        bestj -= 1
        if not self.globalAlign:
            besti, bestj = numpy.argwhere(scores == numpy.amax(scores))[-1]
        else:
            # still have to find best final index to start from
            terminalIndices = [index for (index, pidx) in enumerate(ni) if self.parent.out_degree(pidx)==0]

            besti = terminalIndices[0] + 1
            bestscore = scores[besti, bestj]
            for i in terminalIndices[1:]:
                score = scores[i+1, bestj]
                if score > bestscore:
                    bestscore, besti = score, i+1

        matches = []
        strindexes = []
        while (self.globalAlign or scores[besti, bestj] > 0) and not(besti == 0 and bestj == 0):
            nexti, nextj = backGrphIdx[besti, bestj], backStrIdx[besti, bestj]
            curstridx, curnodeidx = self.childidx_order[bestj-1], self.nodeIndexToID[besti-1]

            name_aligned = True
            if curstridx is not None and curnodeidx is not None:
                name_aligned = self.child.nodes[curstridx]['attr']['op_type']==self.parent.nodes[curnodeidx]['attr']['op_type']
            name_aligned = name_aligned and (nextj != bestj and nexti != besti)

            strindexes.append(curstridx if name_aligned else None)
            matches.append(curnodeidx if name_aligned else None)

            besti, bestj = nexti, nextj

        strindexes.reverse()
        matches.reverse()

        return strindexes, matches, bestscore

# from networkx.algorithms.isomorphism import DiGraphMatcher
# print(list(DiGraphMatcher(parent, child).subgraph_isomorphisms_iter()))

def get_model_zoo(path):
    return [os.path.join(path, x) for x in os.listdir(path) if os.path.isfile(os.path.join(path, x)) and '.onnx' in x]

def mapping_func(parent_file, child_graph):
    parent, parent_onnx = load_model_meta(parent_file)
    opt = MatchingOperator(parent=parent)
    mappings, score = opt.get_mappings(child=child_graph)

    # print(opt.alignmentStrings()[0])
    # print("\n\n")
    # print(opt.alignmentStrings()[1])
    # print("\n\n")
    # print(opt.graphStrings()[0])
    # print("\n\n")
    # print(opt.graphStrings()[1])
    return (parent, mappings, score)

def main():
    start_time = time.time()
    num_of_processes = 16

    #zoo_path = './temp_zoo'
    zoo_path = './zoo'
    model_zoo = get_model_zoo(zoo_path)

    # create multiple process to handle model zoos
    #child, child_onnx = load_model_meta('./temp_zoo/resnet50oppo1.onnx')
    child, child_onnx = load_model_meta('./zoo/wide_resnet50_2.onnx')

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

main()
