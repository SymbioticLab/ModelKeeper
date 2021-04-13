#include <iostream>
#include <time.h>
#include <string>
#include "matcher.hpp"

using namespace std;

Matcher* create_matcher(){
    return new Matcher;
}

double get_score(Matcher* t, char* file_path){
	string path_str(file_path);
    return t->gen_mapping(path_str);
}

int main(){
	clock_t start_time = clock();
	//string file_path = "graph_meta.json";
	char file_path[] = "graph_meta.json";
	Matcher* mapper = create_matcher();
	double score = 0;

	score = get_score(mapper, file_path);
	// Matcher mapper;
	// double score = mapper.gen_mapping(file_path);
	cout << score << endl;
	cout << (clock() - start_time)/1000000.0 << endl;
}
