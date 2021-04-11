#include <iostream>
#include <set>

#include <cassert>

#include "DMSTNode.hpp"
#include "Simulator.hpp"

using namespace std;

// for debug
void DMSTNode::LOG(const char * str)
{
	//cout << nodeID << " " << str << endl;
}

void DMSTNode::init(const vector<pair<int, double> > & initIncommingEdges, 
		Simulator * _pSimulator, int id, int numNode)
{
	incommingEdges = initIncommingEdges;
	nodeID = id;
	incEdge.resize(numNode, -1);
	outEdgeSet.resize(numNode, set<int>());
	neighbor.resize(numNode, set<int>());
	pSimulator = _pSimulator;
	// other variables
	minWeight = INF;
	bestEdge = -1;
	bestNode = -1;
	clusterStem = -1;
	stemEdge = -1;
	waitCount = 0;
	recieve(Order(WAKEUP, set<int>(), 0, 0.0, -1));
}

bool DMSTNode::run()
{
	LOG("RUN");
	bool hasRun = false;
	while(!messageQueue.empty()){
		Order & order = messageQueue.front();
		work(order);
		hasRun = true;
		messageQueue.pop();
	}
	LOG("RUN END");
	return hasRun;
}

void DMSTNode::work(const Order & order)
{
	LOG("WORK");
	switch (order.fID){
		case WAKEUP:
			wakeUp();
			break;
		case CONNECT:
			connect(order.pPSet, order.edge);
			break;
		case MAKEKNOWN:
			makeKnown(order.pPSet, order.edge);
			break;
		case LIST:
			list(order.pPSet, order.edge);
			break;
		case CYCLE:
			cycle(order.edge);
			break;
		case REPORT:
			report(order.pInt, order.pDouble, order.edge);
			break;
		case UPDATE:
			update(order.pInt, order.pDouble, order.edge);
			break;
		default:
			assert(0);
	}
	LOG("WORK END");
}

void DMSTNode::wakeUp()
{
	LOG("WAKEUP");
	knownSet.insert(nodeID);
	minWeight = INF;
	for(int i = incommingEdges.size() - 1; i >= 0; i--){
		if(incommingEdges[i].second < minWeight){
			minWeight = incommingEdges[i].second;
			bestEdge = incommingEdges[i].first;
		}
	}
	update(nodeID, minWeight, -1);
	LOG("WAKEUP END");
}

void DMSTNode::connect(const set<int> & nodeSet, int edge)
{
	LOG("CONNECT");
	neighbor[edge] = nodeSet;
	makeKnown(knownSet, edge);
	LOG("CONNECT END");
}

void DMSTNode::makeKnown(const set<int> & nodeSet, int edge)
{
	LOG("MAKEKONWN");
	set<int> sendSet;
	setMinus(nodeSet, neighbor[edge], sendSet);
	for(set<int>::iterator it = sendSet.begin(); it != sendSet.end(); it++){
		outEdgeSet[*it].insert(edge);
	}
	if(sendSet.size()){
		send(Order(LIST, sendSet, 0, 0.0, edge));
	}
	if(newInternalSet.find(edge) == newInternalSet.end() && hasCross(
		nodeSet, neighbor[edge])){
		newInternalSet.insert(edge);
		send(Order(CYCLE, set<int>(), 0, 0.0, edge));
	}
	if(maxElement(knownSet) > maxElement(neighbor[edge])){
		waitCount ++;
	}
	LOG("MAKEKNOWN END");
}

void DMSTNode::list(const set<int> & nodeSet, int edge)
{
	LOG("LIST");
	setMerge(knownSet, nodeSet);
	for(set<int>::iterator it = nodeSet.begin(); it != nodeSet.end(); it++){
		incEdge[*it] = edge;
		outEdgeSet[*it].clear();
	}
	for(set<int>::iterator it = outEdgeSet[clusterStem].begin(); 
			it != outEdgeSet[clusterStem].end(); it++){
		makeKnown(nodeSet, *it);
	}
	LOG("LIST END");
}

void DMSTNode::cycle(int edge)
{
	LOG("CYCLE");
	for(set<int>::iterator it = internalSet.begin(); it != internalSet.end(); it++){
		if(outEdgeSet[clusterStem].find(*it) == outEdgeSet[clusterStem].end())
			continue;
		send(Order(CYCLE, set<int>(), 0, 0.0, *it));
		waitCount++;
	}
	for(int i = incommingEdges.size() - 1; i >= 0; i--){
		int j = incommingEdges[i].first;
		double edgeWeight = incommingEdges[i].second;
		if(knownSet.find(j) == knownSet.end() && edgeWeight < minWeight){
			minWeight = edgeWeight;
			bestEdge = j;
		}
	}
	report(nodeID, minWeight, -1);
	LOG("CYCLE END");
}

void DMSTNode::report(int node, double bestWeight, int edge)
{
	LOG("REPORT");
	if(bestWeight <= minWeight){
		minWeight = bestWeight;
		bestNode = node;
	}
	waitCount --;
	if(!waitCount){
		if(nodeID == clusterStem && clusterID == maxElement(knownSet)){
			// update from the new stem of the cycle detected, i.e. the 
			// node has minimum incomming edge.
			update(bestNode, minWeight, -1);
		}
		else{
			send(Order(REPORT, set<int>(), bestNode, minWeight, stemEdge));
		}
	}
	LOG("REPORT END");
}

void DMSTNode::update(int newClusterStem, double bestWeight, int edge)
{
	if(bestWeight == INF) return;
	LOG("UPDATE");
	set<int> tmp;
	setIntersection(outEdgeSet[newClusterStem], newInternalSet, tmp);
	if(incEdge[newClusterStem] != -1){
		tmp.insert(incEdge[newClusterStem]);
	}
	for(set<int>::iterator it = tmp.begin(); it != tmp.end(); it++){
		if(*it == edge) continue;
		if(*it == newClusterStem) continue;
		send(Order(UPDATE, set<int>(), newClusterStem, bestWeight, *it));
	}
	clusterStem = newClusterStem;
	clusterID = maxElement(knownSet);
	internalSet = newInternalSet;
	minWeight = INF;
	waitCount = 1;
	for(int i = incommingEdges.size() - 1; i >= 0; i--){
		incommingEdges[i].second -= bestWeight;
	}
	if(clusterStem == nodeID){
		send(Order(CONNECT, knownSet, 0, 0.0, bestEdge));
		stemEdge = bestEdge;
	}
	else{
		stemEdge = incEdge[clusterStem];
	}
	LOG("UPDATE END");
}

void DMSTNode::send(const Order & order)
{
	LOG("SEND");
	// send by simulator
	assert(order.edge != -1);
	pSimulator->send(order, nodeID);
}

void DMSTNode::recieve(const Order & order)
{
	LOG("RECIEVE");
	messageQueue.push(order);
}

int DMSTNode::getBestEdge(int root)
{
	return incEdge[root];
}

set<int> & DMSTNode::setMinus(const set<int> & s1, const set<int> & s2, set<int> & result){
	result.clear();
	for(set<int>::iterator it = s1.begin(); it != s1.end(); it++){
		if(s2.find(*it) == s2.end()){
			result.insert(*it);
		}
	}
	return result;
}

set<int> & DMSTNode::setMerge(set<int> & target, const set<int> & source)
{
	for(set<int>::iterator it = source.begin(); it != source.end(); it++){
		target.insert(*it);
	}
	return target;
}

bool DMSTNode::hasCross(const set<int> & s1, const set<int> & s2)
{
	for(set<int>::iterator it = s1.begin(); it != s1.end(); it++){
		if(s2.find(*it) != s2.end()) return true;
	}
	return false;
}

int DMSTNode::maxElement(const set<int> & st)
{
	assert(st.size());
	return *(st.begin());
}

set<int> & DMSTNode::setIntersection(const set<int> & s1, const set<int> & s2, set<int> & res)
{
	res.clear();
	for(set<int>::iterator it = s1.begin(); it != s1.end(); it++){
		if(s2.find(*it) != s2.end())
			res.insert(*it);
	}
	return res;
}

