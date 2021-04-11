#ifndef __DMSTNode__
#define __DMSTNode__

#include <set>
#include <queue>
#include <vector>

using namespace std;

const double INF = 1e20;

enum FunctionID { WAKEUP, CONNECT, MAKEKNOWN, LIST, CYCLE, REPORT, UPDATE};
struct Order{
	FunctionID fID;
	set<int> pPSet;
	int pInt;
	double pDouble;
	int edge;
	Order(FunctionID _fID, const set<int> & _set, int _int, double _double, 
			int _edge): fID(_fID), pPSet(_set), pInt(_int), pDouble(_double),
			edge(_edge){}
};

class DMSTNode{
private:
	set<int> knownSet;
	int nodeID;
	set<int> newInternalSet;
	set<int> internalSet;
	// incEdge[i] : the incomming edge when this node belong to the DMST rooted as i
	vector<int> incEdge;
	vector<set<int> > outEdgeSet;
	double minWeight;
	// incomming edge has best weight
	int bestEdge;
	// best stem node found so far
	int bestNode;
	vector<pair<int, double> > incommingEdges;
	vector<set<int> > neighbor;
	int waitCount;
	int clusterStem;
	int clusterID;
	int stemEdge;
	class Simulator * pSimulator;
	queue<Order> messageQueue;
public:
	// TODO numNode is essential
	void init(const vector<pair<int, double> > & initIncommingEdges, 
			Simulator * _pSimulator, int id, int numNode);
	bool run();
	int getBestEdge(int root);
	void recieve(const Order & order);
private:
	void work(const Order & order);
	void wakeUp();
	void connect(const set<int> & nodeSet, int edge);
	void makeKnown(const set<int> & nodeSet, int edge);
	void list(const set<int> & nodeSet, int edge);
	void cycle(int edge);
	void report(int node, double bestWeight, int edge);
	void update(int newClusterStem, double bestWeight, int edge);
	void send(const Order & order);
private:
	set<int> & setMinus(const set<int> & s1, const set<int> & s2, set<int> & res);
	set<int> & setMerge(set<int> & target, const set<int> & source);
	bool hasCross(const set<int> & s1, const set<int> & s2);
	int maxElement(const set<int> & st);
	set<int> & setIntersection(const set<int> & s1, const set<int> & s2, set<int> & res);
private:
	void LOG(const char * str);
};

#endif
