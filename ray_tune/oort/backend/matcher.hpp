#ifndef __MATCHER__
#define __MATCHER__

#include <vector>
#include <map>
#include <string>

using namespace std;

struct node_pair {
	double val;
	int parentidx;
	int childidx;
	int opt;

	node_pair(double v, int p, int c, int o) :val(v), parentidx(p), childidx(c), opt(o) {}
};

struct Node
{	
	int idx;
	string type;
	vector<int> shape;
	vector<int> parents;
};

class Matcher{

private:
	int len_parent;
	int len_child;
	bool dump_mapping;
	string json_path;

	double **scores;
	map<string, vector<int> > backParentIdx;
	map<string, vector<int> > backChildIdx;

	vector<Node> parent_nodes;
	vector<Node> child_nodes;

	vector<long long> parent_parameters;
	vector<long long> child_parameters;

public:

	double gen_mapping(string file_path, bool dump_mapping);

	// merge k sorted list
	inline double merge_branch_mapping(vector<vector<node_pair> > lists, vector<int> & parent_list, vector<int> & child_list);
	inline double cal_score(Node parent_node, Node child_node);

	void read_io(string file_path);
	void align_child_parent();
	void init_score();
	void dump_trace();

	string encode_hash(int i, int j);
};

#endif
