#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include <time.h>
#include <climits>
#include <set>

#include "json.hpp"
#include "matcher.hpp"

using json=nlohmann::json;

#define MATCH 1
#define MISMATCH -1
#define DEL -1
#define INS 0

#define _matchscore 1
#define _mismatchscore -1 // may miss better skip/insert, so take (ins+del)/2.0
#define ins_gap -0.25  // insert identity mapping, insert too many is bad. so slightly negative
#define del_gap -1  // lose all information

using namespace std;


char* Matcher::gen_mapping(string json_s, bool dump){
    dump_mapping = dump;

    read_io(json_s);
    init_score();
    align_child_parent();

    json json_ans;

    if (dump_mapping) {
        json_ans = {
            {"score", scores[len_parent][len_child]},
            {"backParentIdx", backParentIdx},
            {"backChildIdx", backChildIdx}
        };
    } else {
        json_ans = {{"score", scores[len_parent][len_child]}};
    }

    string ans_str = json_ans.dump();

    char * writable = new char[ans_str.size() + 1];
    copy(ans_str.begin(), ans_str.end(), writable);
    writable[ans_str.size()] = '\0'; // don't forget the terminating 0

    return writable;
}

string Matcher::encode_hash(int i, int j){
    return to_string(i)+"_"+to_string(j);
}


void parse_node_info(json metadata, vector<Node> & node_list){
    json opt_list = metadata["opts"];
    json dim_list = metadata["dims"];
    json parent_list = metadata["parents"];

    for (int i=0; i < opt_list.size(); ++i) {
        vector<int> shapes;
        vector<int> parents;

        for (int j=0; j < dim_list[i].size(); ++j) {
            shapes.push_back(dim_list[i][j].get<int>());
        }

        for (int j=0; j < parent_list[i].size(); ++j) {
            parents.push_back(parent_list[i][j].get<int>());
        }

        Node temp = {i, opt_list[i].get<string>(), shapes, parents};
        node_list.push_back(temp);
    }
}


void Matcher::read_io(string json_s){
    json root = json::parse(json_s);

    len_parent = root.at("len_parent").get<int>();
    len_child = root.at("len_child").get<int>();

    parse_node_info(root["parent"], parent_nodes);
    parse_node_info(root["child"], child_nodes);
}



void Matcher::init_score(){
    scores = new double*[1+len_parent];
    for (int i=0; i < 1+len_parent; ++i) {
        scores[i] = new double[1+len_child]();
    }
    
    double best;

    for (int i=0; i < parent_nodes.size(); ++i) {
        best = INT_MIN;

        for (int j=0; j < parent_nodes[i].parents.size(); ++j) {
            best = max(best, scores[1+parent_nodes[i].parents[j]][0]);
        }
        scores[i+1][0] = (best + del_gap);///parent_nodes[i].parents.size();
    }

    for (int i=0; i < child_nodes.size(); ++i) {
        best = INT_MIN;

        for (int j=0; j < child_nodes[i].parents.size(); ++j){
            best = max(best, scores[0][1+child_nodes[i].parents[j]]);
        }
        scores[0][i+1] = (best + ins_gap);//child_nodes[i].parents.size();
    }

    long long num_parent_param=1;
    // calculate number of parameters
    for (int i=0; i < parent_nodes.size(); ++i) {
        num_parent_param = 1;
        for (int j=0; j < parent_nodes[i].shape.size(); ++j){
            num_parent_param *= parent_nodes[i].shape[j];
        }
        parent_parameters.push_back(num_parent_param);
    }

    for(int i=0; i < child_nodes.size(); ++i) {
        num_parent_param = 1;
        for (int j=0; j < child_nodes[i].shape.size(); ++j){
            num_parent_param *= child_nodes[i].shape[j];
        }
        child_parameters.push_back(num_parent_param);
    }
}

inline double Matcher::merge_branch_mapping(vector<vector<node_pair> > lists, vector<int> & parent_list, vector<int> & child_list){

    priority_queue<pair<double, int> > queue;

    for(int i=0; i < lists.size(); ++i){
        queue.push(make_pair(lists[i][0].val, i));
    }

    vector<int> inbranch_idx(lists.size(), 0);

    double score = 0;
    int branch, inbranch, parent_node;
    set<int> parent_node_set;

    // merge k-sorted lists
    while (queue.size() > 0){
        pair<double, int> temp_pair = queue.top();
        queue.pop();

        branch = temp_pair.second;
        inbranch = inbranch_idx[branch];

        //if (lists[branch][inbranch].opt == MATCH){
        parent_node = lists[branch][inbranch].parentidx;

        // parent is used before, then move to the next inbranch idx in this branch
        if (parent_node_set.find(parent_node) != parent_node_set.end()){
            if (inbranch + 1 < lists[branch].size()){
                inbranch += 1;
                inbranch_idx[branch] = inbranch;
                queue.push(make_pair(lists[branch][inbranch].val, branch));
            }
        } else{
            parent_node_set.insert(parent_node);
            score += lists[branch][inbranch].val;

            if (dump_mapping) {
                parent_list.push_back(lists[branch][inbranch].parentidx);
                child_list.push_back(lists[branch][inbranch].childidx);
            }
        
        }
    }

    return score;
}


inline double Matcher::cal_score(Node parent_node, Node child_node){
    if (parent_node.type != child_node.type || parent_node.shape.size() != child_node.shape.size()) {
        return _mismatchscore;
    } else{
        long long num_parent_param=parent_parameters[parent_node.idx], num_child_param=child_parameters[child_node.idx];
        long long inherited_param=1;

        size_t shape_size = min(parent_node.shape.size(), child_node.shape.size());
        for (int i=0; i < shape_size; ++i){
            inherited_param *= min(parent_node.shape[i], child_node.shape[i]);
        }

        double match_score = inherited_param;
        if (num_parent_param > num_child_param) {
            match_score = inherited_param/num_parent_param/1.5; // lost information of the parent
        } else {
            match_score = inherited_param/num_child_param; // padding too many ones
        }
        //double match_score = inherited_param/max(num_parent_param, num_child_param);

        // lose too much information
        if (match_score > 0.5) {
            return match_score * _matchscore;
        }

        // skip this layer
        return _mismatchscore;
    }
}

bool cmp_function(node_pair i, node_pair j) { return (i.val>=j.val); }

void Matcher::align_child_parent(){

    double match_score = 0, merge_score = 0;
    int cprev = 0, is_match=1, predIndex=0;
    map<string, vector<int> >::iterator iter;
    vector<int> temp_vec;
    bool is_branch;

    // Dynamic Programming
    for (int i=0; i < parent_nodes.size(); ++i){
        Node parent_node = parent_nodes[i];

        for (int j=0; j < child_nodes.size(); ++j){
            Node child_node = child_nodes[j];
            match_score = cal_score(parent_node, child_node);
            is_match = match_score > _mismatchscore ? MATCH:MISMATCH;
            is_branch = (child_node.parents.size() > 1);

            vector<vector<node_pair> > temp_ans;

            // enumerate all branches
            // TODO: optimal solution should apply bipartite algorithm to mix and match 
            for (int k=0; k < child_node.parents.size(); ++k){
                vector<node_pair> temp;
                cprev = child_node.parents[k]; // j

                // insert identity mapping
                temp.push_back(node_pair(scores[i+1][cprev+1] + ins_gap, i+1, cprev+1, INS));

                // add all candidates to a list, pick the best insert to the child
                for (int m=0; m < parent_node.parents.size(); ++m) {
                    predIndex = parent_node.parents[m];
                    temp.push_back(node_pair(scores[predIndex+1][cprev+1] + match_score, predIndex+1, cprev+1, is_match));
                    if (is_branch == false)
                       temp.push_back(node_pair(scores[predIndex+1][j+1] + del_gap, predIndex+1, j+1, DEL)); // skip a child node
                }
                
                sort(temp.begin(), temp.end(), cmp_function);
                temp_ans.push_back(temp);
            }

            // merge branches
            vector<int> parent_list, child_list;

            merge_score = merge_branch_mapping(temp_ans, parent_list, child_list);

            // treat multi branches as a single chain 
            scores[i+1][j+1] = merge_score;///child_node.parents.size();//max(parent_node.parents.size(), child_node.parents.size());

            if (dump_mapping){
                backParentIdx.insert({encode_hash(i+1, j+1), parent_list});
                backChildIdx.insert({encode_hash(i+1, j+1), child_list});
            }
        }
    }

}

extern "C"{
    char* get_matching_score(char json_str[], bool dump_mapping){
        string json_s(json_str);
        Matcher mapper;

        return mapper.gen_mapping(json_s, dump_mapping);
    }
}

