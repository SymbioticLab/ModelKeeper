
import networkx as nx 
from networkx.algorithms.tree.branchings import Edmonds
import time

class GraphOperator(object):
    """Map parent weights to child weights given the mapping index"""

    def __init__(self, ):
        self.graph = None

    def max_spanning_tree(self):
        start_time = time.time()

        tree_opt = Edmonds(self.graph, seed=0)
        mst = tree_opt.find_optimum(style="arborescence")

        topo_order = list(nx.topological_sort(nx.line_graph(mst)))
        sum_weight = sum([mst[u][v]['weight'] for u, v in mst.edges()])

        print(f"max_spanning_tree takes {time.time() - start_time} sec: weight {sum_weight}")
        return topo_order

    def load_graph(self, meta_file):    
        self.graph = nx.DiGraph()

        with open(meta_file) as fin:
            lines = fin.readlines()
            
            [num_nodes, num_edges] = lines[0].strip().split()
            self.graph.add_nodes_from(list(range(int(num_nodes))))

            for line in lines[1:]:
                [src, dst, weight] = line.strip().split()   
                self.graph.add_edge(src, dst, weight=float(weight))



def test():
    graph_opt = GraphOperator()
    graph_opt.load_graph('graph.in')
    print(graph_opt.max_spanning_tree())

test()
