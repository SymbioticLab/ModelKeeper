#include <iostream>
#include <fstream>
#include <vector>

#include "Network.hpp"

using namespace std;

int main(int args, char * argv[])
{
	int n, m;

	if(args < 2){
		cout << "USAGE: " << argv[0] << " " << "test_file" << endl;
		return 1;
	}
	ifstream fin(argv[1]);
	while(fin >> n >> m){
		Network network(n);
		while(m--){
			int x, y;
			double dist;
			fin >> x >> y >> dist;
			network.addEdge(x, y, -dist);
		}
		cout << network.simulateDistribution(0) << endl;	
		cout << network.zhuliu(0) << endl;
	}
	cout << "Done ...";
	return 0;
}
