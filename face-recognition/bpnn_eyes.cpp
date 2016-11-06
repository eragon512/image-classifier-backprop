#include <map>
#include <ctime>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "readpmg.h"
#include "feature.h"
using namespace std;

#define learning_rate 0.3
#define momentum 0.3
#define epochs 60
#define no_input 960
#define no_hidden 20
#define no_output 1

double sigmoid (double x) {
	return 1.0 / (1.0 + exp(-x));
}

//random weight b/w 0 and 1
double random_weight() {
	return (((double)random()/(double)RAND_MAX) * 0.1) - 0.05;
	//return ( ( (double)(rand()%100)+1)/100 * 2 * 0.05 ) - 0.05;
	//return 0.6;
}

double Wxh[no_input+1][no_hidden];
double Why[no_hidden+1][no_output];

float x[no_input+1],y[no_output];

float h_out[no_hidden+1];
float o_out[no_output];
float o_err[no_output];
float h_err[no_hidden+1];

float deltaWxh_prev[no_input+1][no_hidden] = {0}; //adjustments to weights between inputs x and hidden nodes
float deltaWhy_prev[no_hidden+1][no_output] = {0}; //adjustments to weights between hidden nodes and output y

void initialise_weights();
void feed_forward();
double get_error();
void adjust_weights();
void back_propagation();

void print_weights();
void print_weight_changes();
void print_layers();

void initialise_weights() {
	int i,h,o;
	for(i=0;i<=no_input;++i) {
		for(h=0;h<no_hidden;++h) {
			Wxh[i][h] = random_weight();
		}
	}
	for(h=0;h<=no_hidden;++h) {
		for(o=0;o<no_output;++o) {
			Why[h][o] = random_weight();
		}
	}
}

void feed_forward() {
	int i,h,o;

	x[no_input] = -1.0;
	for(h=0;h<no_hidden;++h) {
		double tmp=0.0;
		for(i=0;i<=no_input;++i) {
			tmp += Wxh[i][h] * x[i];
		}
		h_out[h] = sigmoid(tmp);
	}

	h_out[no_hidden] = -1.0;
	for(o=0;o<no_output;++o) {
		double tmp=0.0;
		for(h=0;h<=no_hidden;++h) {
			tmp += Why[h][o] * h_out[h];
		}
		o_out[o] = sigmoid(tmp);
	}
}

double get_error() {
	double error=0,error_sum=0;
	int o,h;
	
	for(o=0;o<no_output;++o) {
		o_err[o] = o_out[o] * (1.0 - o_out[o]) * (y[o] - o_out[o]);
		error += (y[o] - o_out[o]) * (y[o] - o_out[o]) * 0.5;
	}
	
	for(h=0;h<=no_hidden;++h) {
		double tmpsum=0.0;
		for(o=0;o<no_output;++o) {
			tmpsum += o_err[o] * Why[h][o];
		}
		h_err[h] = h_out[h] * (1.0 - h_out[h]) * tmpsum;
	}
	//printf("Error: %f\n",error);
	return error;
}

void adjust_weights() {
	int i,h,o;
	for(h=0; h<=no_hidden; h++) {
		for (o=0; o<no_output; o++) {
			double delta = 	(learning_rate * o_err[o] * h_out[h]) + (momentum * deltaWhy_prev[h][o]);
			Why[h][o] += delta;
			deltaWhy_prev[h][o] = delta;
		}
	}

	for(i=0; i<=no_input; i++) {
		for (h=0; h<no_hidden; h++) {
			double delta = (learning_rate * h_err[h] * x[i]) + (momentum * deltaWxh_prev[i][h]);
			Wxh[i][h] += delta;
			deltaWxh_prev[i][h] = delta;
		}
	}
}

void back_propagation() {
	feed_forward();
	get_error();
	adjust_weights();
}

void load_instance(vector<float>& in,vector<float> &out) {
	copy(in.begin(),in.end(),x);
	copy(out.begin(),out.end(),y);
}

double rmse() {
	double mse=0.0;
	for(int o=0;o<no_output;++o) {
		mse += (y[o] - o_out[o]) * (y[o] - o_out[o]);
	}
	mse = (double)mse/(double)no_output;
	double rmse = sqrt(mse);
	return rmse;
}

void print_performance(double avg_rmse,double max_rmse,double accuracy) {
	printf("Average RMSE: %lf\n",avg_rmse);
	printf("Max RMSE: %lf\n",max_rmse);
	printf("Accuracy: %.2lf%%\n",accuracy*100.0);
}

void fetch_dataset(const string& imglistfile,vector< pair< vector<float>,vector<float> > >& bpnn_train) {
	vector<string> imglist= get_imglist(imglistfile);
	for(int q=0;q<imglist.size();++q) {
		bpnn_train.push_back(make_pair(read_pmg_normalised(imglist[q]),get_eyes_vector(imglist[q])));
	}
}

void train_network(const string &imglistfile) {
	vector< pair< vector<float>,vector<float> > > bpnn_train;
	fetch_dataset(imglistfile,bpnn_train);
	
	int len=bpnn_train.size();
	for(int e=0;e<epochs;++e) {
		double avg_rmse=0.0,max_rmse=0.0,accuracy=0.0;
		//random_shuffle(bpnn_train.begin(),bpnn_train.end());
		
		for(int l=0;l<len;++l) {
			load_instance(bpnn_train[l].first,bpnn_train[l].second);
			back_propagation();
			
			double RMSE = rmse();
			avg_rmse += RMSE/(double)len;
			max_rmse = max(RMSE,max_rmse);
			accuracy += classify_eyes(o_out,no_output,y,no_output)/(double)len;

			//evaluate_output();
			//print_weight_changes();
			//print_layers();
		}
		printf(" ------------------ Epoch %d -------------------\n",e+1);
		//print_weights();
		print_performance(avg_rmse,max_rmse,accuracy);
		if(accuracy > 0.999) {
			break;
		}
		//print_layers();
	}
}

void test_network(const string &imglistfile) {
	vector< pair< vector<float>,vector<float> > > bpnn_test;
	fetch_dataset(imglistfile,bpnn_test);
	
	int len=bpnn_test.size();

	printf("---------------- Testing Neural Network -------------\n");
	
	double avg_rmse=0.0,max_rmse=0.0,accuracy=0.0;
	for(int l=0;l<len;++l) {
		load_instance(bpnn_test[l].first,bpnn_test[l].second);
		feed_forward();
			
		double RMSE = rmse();
		avg_rmse += RMSE/(double)len;
		max_rmse = max(RMSE,max_rmse);
		accuracy += classify_eyes(o_out,no_output,y,no_output)/(double)len;
		//evaluate_output();
		//print_weight_changes();
		//print_layers();
	}

	//print_weights();
	print_performance(avg_rmse,max_rmse,accuracy);
	//print_layers();
}

int main() {
	initialise_weights();
	train_network("dataset/lists/straightrnd_train.list");
	test_network("dataset/lists/straightrnd_test1.list");
	test_network("dataset/lists/straightrnd_test2.list");
	return 0;
}

void print_weights() {
	int i,h,o;
	printf("Weights: \n");
	printf("Wxh: \n");
	for (i=0; i<=no_input; i++) {
		for (h=0; h<no_hidden; h++) {
			printf("%f ",Wxh[i][h]);
		}
		printf("\n");
	}
	printf("Why: \n");
	for (h=0; h<=no_hidden; h++) {
		for (o=0; o<no_output; o++) {
			printf("%f ",Why[h][o]);
		}
		printf("\n");
	} /**/
	printf("\n");
}

void print_weight_changes() {
	int i,h,o;
	printf("Weight Changes: \n");
	printf("deltaWxh: \n");
	for (i=0; i<=no_input; i++) {
		for (h=0; h<no_hidden; h++) {
			printf("%f ",deltaWxh_prev[i][h]);
			//Wxh[i][h] = 0;
		}
		printf("\n");
	}
	printf("deltaWhy: \n");
	for (h=0; h<=no_hidden; h++) {
		for (o=0; o<no_output; o++) {
			printf("%f ",deltaWhy_prev[h][o]);
		}
		printf("\n");
	} /**/
	printf("\n");
}

void print_layers() {
	int i,h,o;
	printf("Layers: \n");
	printf("Input Layer: ");
	for (i=0; i<=no_input; i++) {
		printf("%f ",x[i]);
	}
	printf("\nHidden Layer: ");
	for (h=0; h<=no_hidden; h++) {
		printf("%f ",h_out[h]);
	}
	printf("\nOutput Layer: ");
	for (o=0; o<no_output; o++) {
		printf("%f ",o_out[o]);
	}
	printf("\n\n");
}