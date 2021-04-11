#include <iostream>
#include <vector>

#include "Network.hpp"
#include "Simulator.hpp"

using namespace std;

Network::Network(int n)
{
	numNode = n;
	vector<double> tmp(numNode, INF);
	for(int i = 0; i < numNode; i++){
		graph.push_back(tmp);
	}
}

void Network::addEdge(int i, int j, double weight)
{
	if(i == j) return;
	if(weight < graph[i][j])
		graph[i][j] = weight;
}

double Network::simulateDistribution(int root)
{
	Simulator simulator(graph);
	simulator.run();
	return simulator.getMST(root);
}

int Network::findFather(const vector<int> & father, int p)
{
	while(father[p] != p)
		p = father[p];
	return p;
}

double Network::zhuliu(int root)
{
	vector<vector<pair<int, double> > > incEdge(numNode, vector<pair<int, double> >());
	for(int i = 0; i < numNode; i++){
		for(int j = 0; j < numNode; j++)
			if(graph[j][i] != INF){
				incEdge[i].push_back(make_pair(j, graph[j][i]));
			}
	}
	vector<int> pre(numNode, -1);
	vector<double> dist(numNode);
	vector<int> father;
	for(int i = 0; i < numNode; i++)
		father.push_back(i);
	bool find = true;
	double res = 0.0;
	while(find){
		find = false;
		for(int i = 0; i < numNode; i++){
			if(i == root) continue;
			if(father[i] != i) continue;
			if(pre[i] == -1){
				double minW = INF;
				for(int k = incEdge[i].size() - 1; k >= 0; k--){
					if(incEdge[i][k].second < minW){
						minW = incEdge[i][k].second;
						pre[i] = findFather(father, incEdge[i][k].first);
					}
				}
				for(int k = incEdge[i].size() - 1; k >= 0; k--){
					incEdge[i][k].second -= minW;
				}
				if(minW != INF)
					res += minW;
			}
		}
		vector<int> used(numNode, -1);
		for(int i = 0; i < numNode; i++){
			if(father[i] != i) continue;
			if(used[i] != -1) continue;
			int now = i;
			int start = -1;
			while(now != -1){
				if(used[now] == i){
					start = now;
					break;
				}
				used[now] = i;
				now = pre[now];
			}
			if(start != -1){
				now = start;
				do{
					used[now] = start;
					father[now] = start;
					now = pre[now];
				}while(now != start);
				find = true;
			}
		}
		for(int i = 0; i < numNode; i++){
			if(father[i] != i) continue;
			for(int j = 0; j < numNode; j++)
				dist[j] = INF;
			for(int j = 0; j < numNode; j++){
				if(father[j] == i){
					for(int k = incEdge[j].size() - 1; k >= 0; k--){
						int p = findFather(father, incEdge[j][k].first);
						if(p == i) continue;
						double nowD = incEdge[j][k].second;
						if(nowD < dist[p])
							dist[p] = nowD;
					}
				}
			}
			incEdge[i].clear();
			for(int j = 0; j < numNode; j++){
				if(dist[j] != INF)
					incEdge[i].push_back(make_pair(j, dist[j]));
			}
			if(pre[i] != -1 && findFather(father, pre[i]) == i)
				pre[i] = -1;
		}
	}
	int numEdge = 0;
	for(int i = 0; i < numNode; i++)
		if(pre[i] != -1)
			numEdge++;
	if(numEdge != numNode - 1)
		res = -1.0;
	return res;
}
