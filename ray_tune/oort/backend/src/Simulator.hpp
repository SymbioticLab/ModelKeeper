#ifndef __SIMULATOR__
#define __SIMULATOR__

#include <vector>

#include "DMSTNode.hpp"

using namespace std;

class Simulator{
private:
	int numNode;
	vector<vector<double> > graph;
	vector<DMSTNode> nodes;
public:
	Simulator(const vector<vector<double> > _graph);
	void run();
	double getMST(int root);
	void send(const Order & order, int from);
private:
	void build();
};

#endif
