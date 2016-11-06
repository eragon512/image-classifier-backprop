#ifndef FEATURE_H_   /* Include guard */
#define FEATURE_H_

#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

std::vector<std::string> get_imglist(const std::string& imglistfile) {
	std::vector<std::string> imglist;
	std::string imgfile;
	std::fstream fin;
	fin.open(imglistfile);
	while(fin >> imgfile) {
		imglist.push_back(imgfile);
	}
	fin.close();
	return imglist;
}

void tokenize(const std::string& str,std::vector<std::string>& tokens,const std::string& delimiters = " ")
{
    // Skip delimiters at beginning.
    std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
    // Find first "non-delimiter".
    std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

    while (std::string::npos != pos || std::string::npos != lastPos)
    {
        // Found a token, add it to the std::vector.
        tokens.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiters, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiters, lastPos);
    }
}

std::string get_filename(const std::string& filepath) {
	std::size_t found = filepath.find_last_of("/\\");
	//std::cout<<filepath.substr(found+1)<<"\n";
	return filepath.substr(found+1);
}

std::vector<std::string> extract_features(const std::string& filepath) {
	std::vector<std::string> features;
	std::string filename = get_filename(filepath);
	tokenize(filename,features,"_");
	return features;
}

std::map< std::string,std::vector<float> > mp_eyes {
	{"open",std::vector<float>{0}},
	{"sunglasses",std::vector<float>{1}}
};

std::map< std::string,std::vector<float> > mp_names {
	{"an2i",	std::vector<float>{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
	{"at33",	std::vector<float>{0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
	{"boland",	std::vector<float>{0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
	{"bpm",		std::vector<float>{0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
	{"ch4f",	std::vector<float>{0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
	{"cheyer",	std::vector<float>{0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0}},
	{"choon",	std::vector<float>{0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0}},
	{"danieln",	std::vector<float>{0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0}},
	{"glickman",std::vector<float>{0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0}},
	{"karyadi",	std::vector<float>{0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0}},
	{"kawamura",std::vector<float>{0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0}},
	{"kk49",	std::vector<float>{0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0}},
	{"megak",	std::vector<float>{0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0}},
	{"mitchell",std::vector<float>{0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0}},
	{"night",	std::vector<float>{0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0}},
	{"phoebe",	std::vector<float>{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}},
	{"saavik",	std::vector<float>{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0}},
	{"steffi",	std::vector<float>{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0}},
	{"sz24",	std::vector<float>{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0}},
	{"tammo",	std::vector<float>{0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}}
};

std::map< std::string,std::vector<float> > mp_poses {
	{"straight",std::vector<float>{1,0,0,0}},
	{"up",		std::vector<float>{0,1,0,0}},
	{"left",	std::vector<float>{0,0,1,0}},
	{"right",	std::vector<float>{0,0,0,1}}
};



std::string get_eyes(const std::string& filepath) {
	std::vector<std::string> features = extract_features(filepath);
	return features[3];
}

std::vector<float> get_eyes_vector(const std::string& filepath) {
	std::string eyes = get_eyes(filepath);
	return mp_eyes[eyes];
}

bool classify_eyes(float o_out[],int out_size,float y[],int y_size) {
	if(y_size != out_size)
		return false;
	else if(y[0] == 0 && o_out[0] < 0.1)
		return true;
	else if(y[0] == 1 && o_out[0] > 0.9)
		return true;
	return false;
}

std::string get_name(const std::string& filepath) {
	std::vector<std::string> features = extract_features(filepath);
	return features[0];
}

std::vector<float> get_name_vector(const std::string& filepath) {
	std::string name = get_name(filepath);
	//std::cout<<name<<"\n";
	if(mp_names[name].empty()) {
		std::cerr<<"Invalid Name "<<name<<"\n";
		exit(1);
	}
	return mp_names[name];
}

bool classify_name(float o_out[],int out_size,float y[],int y_size) {
	if(y_size != out_size)
		return false;

	int max_ind = 0;
	float max_o=0.0;
	for(int o=0;o<out_size;++o) {
		if(max_o < o_out[o]) {
			max_o = o_out[o];
			max_ind = o;
		}
		//std::cout<<o_out[o]<<" ";
	}

	//std::map< std::string,std::vector<float> >::iterator it;
	//for(int i=0,it=mp.begin();it!=mp.end() && i<max_ind;++i,++it);
	//printf("\nPredicted Name: %d, Preciction Strength: %f\n",max_ind,max_o);

	if(max_o > 0.9 && y[max_ind] == 1)
		return true;

	return false;
}

std::string get_pose(const std::string& filepath) {
	std::vector<std::string> features = extract_features(filepath);
	return features[1];
}

std::vector<float> get_pose_vector(const std::string& filepath) {
	std::string pose = get_pose(filepath);
	//std::cout<<name<<"\n";
	if(mp_poses[pose].empty()) {
		std::cerr<<"Invalid Pose\n";
		exit(1);
	}
	return mp_poses[pose];
}

bool classify_pose(float o_out[],int out_size,float y[],int y_size) {
	if(y_size != out_size)
		return false;

	int max_ind = 0;
	float max_o=0.0;
	for(int o=0;o<out_size;++o) {
		if(max_o < o_out[o]) {
			max_o = o_out[o];
			max_ind = o;
		}
		//std::cout<<o_out[o]<<" ";
	}

	//std::map< std::string,std::vector<float> >::iterator it;
	//for(int i=0,it=mp.begin();it!=mp.end() && i<max_ind;++i,++it);
	//printf("\nPredicted Name: %d, Preciction Strength: %f\n",max_ind,max_o);

	if(max_o > 0.9 && y[max_ind] == 1)
		return true;

	return false;
}

std::string get_expression(const std::string& filepath) {
	std::vector<std::string> features = extract_features(filepath);
	return features[2];
}

#endif // FEATURE_H_