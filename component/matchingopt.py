
import onnx
import numpy
import igraph
from igraph import *
import time
import collections
import functools

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
        elif 'weight' in _input:
            layer_name = _input
            break
        elif 'bias' in _input:
            layer_name = _input
            break

    return input_nodes, layer_name

def get_tensor_shapes(model_graph):

    node_shapes = dict()

    if model_graph.initializer:
        print("Load from initializer")
        for init in model_graph.initializer:
            node_shapes[init.name] = tuple(init.dims)
    else:
        print("Load from input")
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
    que = collections.deque()

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
            node = que.pop()
            nodes_by_layer[-1].append(node)

            for s in graph.successors(node):
                in_degrees[s] -= 1

                if in_degrees[s] == 0:
                    del in_degrees[s]
                    que.append(s)

    assert(len(in_degrees) == 0)

    #print([[graph.vs[v]['attribution']['name'] for v in vs] for vs in nodes_by_layer])
    return nodes_by_layer


def clip(var):
    if var < -1.: return -1.
    if var > 1.: return 1.
    return var

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
        self.parentidx_order = self.parent.topological_sorting()

    def align_child(self, child):
        self.child = child
        self.matchidxs = self.parentidxs = None
        self.childidx_order = self.child.topological_sorting()
        matches = self.alignChildToParent()
        self.matchidxs, self.parentidxs, self.match_score = matches

    def alignmentStrings(self):
        return ("\t".join([self.child.vs[i]['attr']['name'] if i is not None else "-" for i in self.matchidxs]),
                "\t".join([self.parent.vs[j]['attr']['name'] if j is not None else "-" for j in self.parentidxs]),
                self.match_score)


    def matchscore(self, parent_opt, child_opt):
        if parent_opt['op_type'] != child_opt['op_type']:
            return self._mismatchscore
        else:
            # not trainable nodes
            if parent_opt['dims'] is None and child_opt['dims'] is None:
                return self._matchscore

            # diff number of parameters
            num_param_p = numpy.prod(parent_opt['dims'])
            num_param_c = numpy.prod(child_opt['dims'])

            match_score = clip(1.-abs(num_param_p-num_param_c)/max(num_param_p, num_param_c)) * self._matchscore

            return match_score

    def get_max(self, candidates):
        ans = 0

        for i in range(1, len(candidates)):
            if (candidates[ans][0], candidates[ans][-1]) <= (candidates[i][0], candidates[i][-1]):
                ans = i
        return candidates[ans]

    def alignChildToParent(self):
        """Align node to parent, following same approach as smith waterman
        example"""
        nodeIDtoIndex, nodeIndexToID, scores, backStrIdx, backGrphIdx = \
                                    self.initializeDynamicProgrammingData(self.parentidx_order)

        # Dynamic Programming
        for i, pidx in enumerate(self.parentidx_order):
            node = self.parent.vs[pidx]
            pbase = node['attr']

            for j, cidx in enumerate(self.childidx_order):
                cnode = self.child.vs[cidx]
                sbase = cnode['attr']
                candidates = [(scores[i+1, j] + self._gap, i+1, j, "INS")]
                # add all candidates to a list, pick the best
                # insert to the child
                for predIndex in self.prevIndices(node, nodeIDtoIndex):
                    candidates += [(scores[predIndex+1, j+1] + self._gap, predIndex+1, j+1, "DEL")]
                    candidates += [(scores[predIndex+1, j] + self.matchscore(sbase, pbase), predIndex+1, j, "MATCH")]

                scores[i+1, j+1], backGrphIdx[i+1, j+1], backStrIdx[i+1, j+1], movetype = self.get_max(candidates)

                if not self.globalAlign and scores[i+1, j+1] < 0:
                    scores[i+1, j+1] = 0.
                    backGrphIdx[i+1, j+1] = -1
                    backStrIdx[i+1, j+1] = -1

        return self.backtrack(scores, backStrIdx, backGrphIdx, nodeIndexToID, self.parentidx_order)

    def prevIndices(self, node, nodeIDtoIndex):
        """Return a list of the previous dynamic programming table indices
           corresponding to predecessors of the current node."""
        prev = []
        for edge in node.in_edges():
            prev.append(nodeIDtoIndex[edge.source])

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
        l1 = len(self.parent.vs)
        l2 = len(self.child.vs)

        nodeIDtoIndex = {}
        nodeIndexToID = {-1: None}
        # generate a dict of (nodeID) -> (index into nodelist (and thus matrix))
        for (index, nidx) in enumerate(ni):
            node = self.parent.vs[nidx]
            nodeIDtoIndex[node.index] = index
            nodeIndexToID[index] = node.index

        # Dynamic Programming data structures; scores matrix and backtracking
        # matrix
        scores = numpy.zeros((l1+1, l2+1), dtype=numpy.int)

        # initialize insertion score
        # if global align, penalty for starting at head != 0
        if self.globalAlign:
            scores[0, :] = numpy.arange(l2+1)*self._gap

            for (index, nidx) in enumerate(ni):
                node = self.parent.vs[nidx]
                prevIdxs = self.prevIndices(node, nodeIDtoIndex)
                best = scores[prevIdxs[0]+1, 0]
                for prevIdx in prevIdxs:
                    best = max(best, scores[prevIdx+1, 0])
                scores[index+1, 0] = best + self._gap

        # backtracking matrices
        backStrIdx = numpy.zeros((l1+1, l2+1), dtype=numpy.int)
        backGrphIdx = numpy.zeros((l1+1, l2+1), dtype=numpy.int)

        return nodeIDtoIndex, nodeIndexToID, scores, backStrIdx, backGrphIdx

    def backtrack(self, scores, backStrIdx, backGrphIdx, nodeIndexToID, ni):
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
            terminalIndices = [index for (index, pidx) in enumerate(ni) if self.parent.outdegree(pidx)==0]

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
            curstridx, curnodeidx = self.childidx_order[bestj-1], nodeIndexToID[besti-1]

            name_aligned = True
            if curstridx is not None and curnodeidx is not None:
                name_aligned = self.child.vs[curstridx]['attr']['op_type']==self.parent.vs[curnodeidx]['attr']['op_type']

            strindexes.insert(0, curstridx if nextj != bestj and name_aligned else None)
            matches.insert(0, curnodeidx if nexti != besti and name_aligned else None)

            besti, bestj = nexti, nextj

        return strindexes, matches, bestscore

skip_opts = {'Constant'}

def load_model_meta(meta_file='sample'):
    start_time = time.time()
    # meta file is rather small
    onnx_model = onnx.load(meta_file+".onnx")
    model_graph = onnx_model.graph

    # record the shape of each weighted nodes
    node_shapes = get_tensor_shapes(model_graph)

    # construct the computation graph and align their attribution
    nodes = [n for n in onnx_model.graph.node if n.op_type not in skip_opts]
    graph = igraph.Graph(directed=True)

    node_ids = dict()
    edge_source = collections.defaultdict(list)

    opt_dir = collections.defaultdict(int)
    for idx, node in enumerate(nodes):
        input_nodes, trainable_weights = split_inputs(node.input)
        opt_dir[node.op_type] += 1

        # add new nodes to graph
        attr = {
            'dims': None if not trainable_weights else node_shapes[trainable_weights],
            'op_type': node.op_type,
            'name': node.name if node.name else str(node.op_type)+str(opt_dir[node.op_type]),
        }
        graph.add_vertex(name=idx, attr=attr)

        # add edges
        for input_node in input_nodes:
            for s in edge_source[input_node]:
                graph.add_edge(source=s, target=idx)

        # register node 
        for out_node in node.output:
            edge_source[out_node].append(idx)

    print('\nLoad {} takes {} sec \n'.format(meta_file, time.time() - start_time))

    return graph

def main():
    start_time = time.time()
    parent = load_model_meta('shufflenet_v2_x1_0')
    child = load_model_meta('vgg19')

    opt = MatchingOperator(parent=parent)
    opt.align_child(child=child)


    print(opt.alignmentStrings()[0])
    print("\n")
    print(opt.alignmentStrings()[1])
    print("\n")
    print(opt.alignmentStrings()[2])

    print("Match takes {} sec".format(time.time() - start_time))

main()
