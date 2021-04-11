#ifndef __NETWORK__
#define __NETWORK__

#include <vector>

using namespace std;

class Network{
private:
	int numNode;
	vector<vector<double> > graph;
private:
	int findFather(const vector<int> & father, int p);
public:
	Network(int n);
	// node id starts from 0
	void addEdge(int i, int j, double weight);
	double simulateDistribution(int root);
	// an implementation of centralized zhu_liu algorithm
	double zhuliu(int root);
};

#endif

