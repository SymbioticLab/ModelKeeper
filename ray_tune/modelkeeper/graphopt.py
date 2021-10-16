import networkx as nx 
from oort_backend.chu_liu_edmonds import chu_liu_edmonds
import time
import pickle
import numpy as np

class GraphOperator(object):
    """Map parent weights to child weights given the mapping index"""

    def __init__(self, ):
        self.graph = None

    def max_spanning_tree(self, score_matrix):
        start_time = time.time()
        heads, tree_score = chu_liu_edmonds(score_matrix)
        print(f"MST takes {time.time() - start_time} sec")

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(list(range(len(heads))))
        scores = 0.

        for i, item in enumerate(heads):
            if item != -1:
                self.graph.add_edge(item, i)
                scores += score_matrix[i, item]

        topo_order = sorted(list(nx.line_graph(self.graph)))

        return topo_order, scores


    def load_graph(self, meta_file): 

        with open(meta_file) as fin:
            lines = fin.readlines()
            
            [num_nodes, num_edges] = lines[0].strip().split()
            score_matrix = np.full((int(num_nodes), int(num_nodes)), np.nan)

            for line in lines[1:]:
                [src, dst, weight] = line.strip().split()
                score_matrix[int(dst), int(src)] = float(weight)

            return score_matrix

    def get_optimal(self, score_matrix):
        return sum(score_matrix.max(axis=0)) - max(score_matrix[:, 0])

    def get_trace_optimal(self, score_matrix):
        score = 0.

        for i in range(1, len(score_matrix)):
            score += max(score_matrix[:i, i])
        return score

    def mst_networkx(self, score_matrix):
        from networkx.algorithms.tree.branchings import Edmonds

        graph = nx.DiGraph()
        graph.add_nodes_from(list(range(len(score_matrix))))

        for i in range(len(score_matrix)):
            for j in range(len(score_matrix)):
                if i != j:
                    graph.add_edge(i, j, weight=score_matrix[i][j])

        tree_opt = Edmonds(graph, seed=0)
        mst = tree_opt.find_optimum(kind="max", style="arborescence")
        sum_weight = [mst[u][v]['weight'] for u, v in mst.edges()]
        return sum(sum_weight)

def test():
    graph_opt = GraphOperator()

    with open('./backend/data/score_graph.pkl', 'rb') as fin:
        score_matrix = pickle.load(fin)

    temp_score_matrix = score_matrix.copy()
    for i in range(len(score_matrix)):
        score_matrix[i][i] = np.nan

    score_matrix = np.transpose(score_matrix)
    _, mst_score = graph_opt.max_spanning_tree(score_matrix)
    global_opt = graph_opt.get_optimal(temp_score_matrix)
    trace_opt = graph_opt.get_trace_optimal(temp_score_matrix)
    nx_opt = graph_opt.mst_networkx(temp_score_matrix)

    print(f"Global Optimal: {global_opt}, MST: {mst_score}, Trace Optimal: {trace_opt}, networkx opt: {nx_opt}")

test()

