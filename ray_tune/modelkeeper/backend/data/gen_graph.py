import networkx as nx 
import time
import sys, random

class ModelPlanner(object):

    def __init__(self, graph):
        self.graph = graph.copy()

    def update_graph(self, graph):
        self.graph = graph.copy()

    def get_mst(self):
        return nx.maximum_spanning_arborescence(self.graph)

    def get_scheduling_order(self, max_spanning_tree):
        return nx.topological_sort(max_spanning_tree)


def faked_graph2():
    graph = nx.DiGraph(name='faked')
    attr1={'dims': [64, 32, 3, 3], 'op_type': 'cov1',}

    graph.add_node(0, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'0'})

    graph.add_node(1, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'1'})
    graph.add_node(2, attr={'dims': [1, 32, 3, 3], 'op_type': 'cov1', 'name':'2'})

    graph.add_node(5, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'5'})
    graph.add_node(3, attr={'dims': [2, 32, 3, 3], 'op_type': 'cov1', 'name':'3'})

    graph.add_node(4, attr={'dims': [5, 32, 3, 3], 'op_type': 'cov1', 'name':'4'})

    graph.add_edge(1, 0, weight=5)
    graph.add_edge(0, 1, weight=1)
    graph.add_edge(1, 2, weight=1)
    graph.add_edge(2, 4, weight=1)
    graph.add_edge(0, 5, weight=1)
    graph.add_edge(5, 3, weight=1)
    graph.add_edge(3, 4, weight=1)

    return graph

def gen_complete_graph(N):
    with open('graph.in', 'w') as fout:
        fout.writelines(str(N)+' ' + str(N**2) + '\n')
        for i in range(N):
            for j in range(N):
                if i != j:
                    fout.writelines(str(i) + ' ' + str(j) + ' '+str(random.randint(1, 100))+'\n')

def test():
    N=500
    graph = nx.complete_graph(N, nx.DiGraph())#faked_graph2()
    planner = ModelPlanner(graph)

    start_time = time.time()
    ans = planner.get_mst()

    print(ans, time.time() - start_time)#planner.get_scheduling_order(ans))
    #print(ans.edges(data=True))

gen_complete_graph(int(sys.argv[1]))
#test()
