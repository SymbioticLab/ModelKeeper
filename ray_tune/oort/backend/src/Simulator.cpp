#include <iostream>

#include "Simulator.hpp"

using namespace std;

Simulator::Simulator(const vector<vector<double> > _graph)
{
	numNode = _graph.size();
	graph = _graph;
	build();
}

void Simulator::build()
{
	nodes.clear();
	nodes.resize(numNode, DMSTNode());
	for(int i = 0; i < numNode; i++){
		vector<pair<int, double> > incEdges;
		for(int j = 0; j < numNode; j++){
			if(graph[j][i] < INF){
				incEdges.push_back(make_pair(j, graph[j][i]));
			}
		}
		nodes[i].init(incEdges, this, i, numNode);
	}
}

void Simulator::send(const Order & order, int from)
{
	int to = order.edge;
	// for debug
	//cout << "simulator send " << from << " " << to << " " << order.fID 
	//	 << " " << order.pInt << " " << order.pDouble << endl;
	Order newOrder = order;
	newOrder.edge = from;
	nodes[to].recieve(newOrder);
}

void Simulator::run()
{
	bool find = true;
	while(find){
		find = false;
		for(int i = 0; i < numNode; i++){
			find |= nodes[i].run();
		}
	}
}

double Simulator::getMST(int root)
{
	double res = 0.0;
	for(int i = 0; i < numNode; i++){
		if(i == root) continue;
		int j = nodes[i].getBestEdge(root);
		if(j == -1) return -1.0;
		//cout << j << " " << i << " " << graph[j][i] << endl;
		res += graph[j][i];
	}
	return res;
}
