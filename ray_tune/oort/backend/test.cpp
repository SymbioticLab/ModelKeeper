#include <iostream>
#include "json.hpp"
#include <string>

using json=nlohmann::json;
using namespace std;

int main(){
	string json_str = "{'nikhil': 1, 'akash': 5, 'manjeet': 10, 'akshat': 15}";
	json second = json::parse(json_str);

	cout << second["nikhil"];
}

