#pragma once
#include <vector>
#include "Neuron.h"
#include <iostream>
#include "Parameter.h"
#include <omp.h>
#include "resource.h"
using namespace std;
//typedef double myreal;

#define IS_MULTITHREADING_USED

class Volume2Vector
{
public:
	Volume2Vector();


	void printParameterInformation();

	void setInformation(std::vector<int>& regular_1D_data, my_int3& dimension, int histogramDimension, int layer1_size,
	                    int distance2, int negative, int random_iteration, int isVoxelBasedHistogramSaved,
	                    int min_count,
	                    int output_mode, myreal sample, myreal alpha, int sample_mode, bool onlyOutOfRangeNegativeUsed,
	                    int classes, int hs, bool is_skip_gram_used);

	void clear();
	void init();
	~Volume2Vector();
	void setRegularVolume(vector<vector<vector<int>>>& regularData, my_int3& dimension);
	int getWordHash(int intword) const;
	int addWordToVocab(int intword);
	int searchVocab(int intword);
	void learnVocab();
	void sortVocab();
	void initUnigramTable();

	void initNet();
	void createBinaryTree();


	void initLambda();
	void trainMultiThreading();
	void initializeNeighborhoodDistribution(long long index);

	void skipGramsMultiTheading(long long index, long long sentance_size);

	
	void save();
	myreal similarity(int index_a, int index_b);

	std::vector<std::vector<myreal>>& getFeatureVector();

	void setFileSaveState(bool is_save_histogram_distribution, bool is_save_similarity_map, bool is_save_volume_vector, bool is_save_cluster);

	void saveVolumeVector();
	void saveHistogramDistribution();
	void saveSimialrityArray();
public:

	std::vector<std::vector<myreal>> feature_vector;
	std::vector<std::vector<int>> histogramBasedVoxel;
	vector<myreal> syn1;
	vector<myreal> syn1neg;
	vector<Neuron> vocab;
	vector<int> table;
	vector<vector<vector<int>>> regularData;
	my_int3 dimension;
	vector<int> vocab_hash;
	vector<myreal> expTable;

	bool is_save_histogram_distribution = false;
	bool is_save_similarity_map = false;
	bool is_save_volume_vector = true;
	bool is_save_cluster = false;

	int vocab_size = 0;
	int layer1_size = 100;				//size of word vector
	int hs = 0;						//Use Hierarchical Softmax; default is 1 (used)
	int negative = 3;				//Number of negative examples; default is 0, common values are 3 - 10 (0 = not used)????
	long long train_words = 0, word_count_actual = 0;
	int debug_mode = 2;				//Set the debug mode (default = 2 = more info during training)
	int min_count = 0;				//This will discard words that appear less than <int> times; default is 0
	int histogramDimension = 256;
	int classes = 3;				//K-means cluster number: default =3;
	int distance2 = 1;
	int random_iteration = 27-1;	//Number of random iterations; default is 26, cube of 3*3*3-1. 
	int isVoxelBasedHistogramSaved = 1;
	int output_mode = 3;				//Output word classes rather than word vectors; default number of classes is 0 (vectors are written), =2 is the cluster, =3 is similarity array
	//Debug 20190513
	myreal sample = 1e-5;			//Set threshold for occurrence of words. Those that appear with higher frequency in the training data
	myreal alpha = 0.05;				//Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
	int sample_mode = 0;			//The mode of sample. 0 is voxel-based and 1 for the value-based.
	bool onlyOutOfRangeNegativeUsed = true;
	myreal starting_alpha;
	int iter = 5;					//Run more training iterations (default 5)???为什么这么少
	const int table_size = 1e8;
	const int vocab_hash_size = 2560000;
	const int exp_table_size = 1000;
	const int max_exp = 6;
	bool is_two_volume_used = false;

public:
	
	vector<myreal> syn0;
	std::vector<std::vector<int>> neighborhood_distribution_array;
	std::vector<myreal> lambda_array;
	std::vector<std::vector<int>> negative_sample_array;
	std::vector<std::vector<myreal>> negative_probability;
	bool is_self_paced_embedding_used = true;
	myreal u = 1; // a threshold parameter, which used to act as a threshold to the sampling probability.
	myreal Lu = 0; //the function acts as L(u) = max(au2+b, 1)
	myreal pram_a = 1e-6, pram_b = 1e-3; // two parameter for L(u) = max(au2+b, 1)
	bool is_skip_gram_used = true;


	bool is_multi_volume_used = false;
	int multivariate_number = 0;
	int single_dimension;
	std::vector<myreal> loss_function_for_batch;
};

