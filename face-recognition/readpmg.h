#ifndef READPMG_H_   /* Include guard */
#define READPMG_H_

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

std::vector<float> read_pmg(std::string filepath) {
	std::string s;     // each line entered by the user
	std::fstream fin;
	int rows,cols,max_val,i;

	fin.open(filepath);
	std::getline(fin,s); //header

	if(s != "P5") {
		std::cerr<<"Invalid PMG format!\n";
		//std::vector<int> V;
		//return &V;
		exit(1);
	}
	
	fin>>rows>>cols>>max_val;
	//std::cout<<rows<<" "<<cols<<" "<<max_val<<"\n";

	getline(fin,s); //null line
	//cout<<"je"<<s<<"\n";

	std::vector<float> V(rows*cols);
	char b;

	for(i=0;i<rows*cols;++i) {
		fin.get(b);
		int q = (int) b;
		V[i] = q;
	}
	fin.close();

	return V;
}

std::vector<float> read_pmg_normalised(std::string filepath) {
	std::string s;     // each line entered by the user
	std::fstream fin;
	int rows,cols,max_val,i;

	fin.open(filepath);
	std::getline(fin,s); //header

	if(s != "P5") {
		std::cerr<<"Invalid PMG format! for "<<filepath<<"\n";
		//std::vector<int> V;
		//return &V;
		exit(1);
	}
	
	fin>>rows>>cols>>max_val;
	//std::cout<<rows<<" "<<cols<<" "<<max_val<<"\n";

	getline(fin,s); //null line
	//cout<<"je"<<s<<"\n";

	std::vector<float> V(rows*cols);
	char b;

	for(i=0;i<rows*cols;++i) {
		fin.get(b);
		int q = (int) b;
		V[i] = (double)q/(double)max_val;
	}
	fin.close();

	return V;
}

#endif // FOO_H_