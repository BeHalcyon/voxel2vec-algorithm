#pragma once
#include "resource.h"
//typedef double myreal;
using namespace hxy;
class Parameter
{

public:
	bool loaded = false;
	int vocab_size = 0;
	int layer1_size = 100;				//size of word vector
	int hs = 1;						//Use Hierarchical Softmax; default is 0 (not used)
	int negative = 0;				//Number of negative examples; default is 0, common values are 3 - 10 (0 = not used)????
	long long train_words = 0;
	int debug_mode = 2;				//Set the debug mode (default = 2 = more info during training)
	int min_count = 0;				//This will discard words that appear less than <int> times; default is 5
	int histogramDimension = 256;	//±Ì ædown sampling volume data histogram dimension
	int classes = 3;
	int distance2 = 1;
	int random_iteration = 27 - 1;	//Number of random iterations; default is 26, cube of 3*3*3-1. 
	int isVoxelBasedHistogramSaved = 1;
	int output_mode = 3;				//Output word classes rather than word vectors; default number of classes is 0 (vectors are written), =2 is the cluster, =3 is similarity array
	//myreal sample = -1e-3;			//Set threshold for occurrence of words. Those that appear with higher frequency in the training data
	myreal sample = 1e-5;
	//myreal alpha = 0.0025;				//Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
	myreal alpha = 0.05;				//Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
	int sample_mode = 0;			//The mode of sample. 0 is voxel-based and 1 for the value-based.
	bool onlyOutOfRangeNegativeUsed = true;
	//Debug 20181218 Added parameters
	int max_code_len = 100;
	int is_uniformed_sapmled_used = 0;
	//Debug 20190513 Added parameters
	bool is_skip_gram_used = true;
	bool is_negative_used = true;
};
