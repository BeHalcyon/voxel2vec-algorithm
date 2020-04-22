#include "Volume2Vector.h"
#include <algorithm>
#include <ctime>
#include <set>
#include <map>


Volume2Vector::Volume2Vector(): starting_alpha(0)
{
	vocab_hash.resize(vocab_hash_size);

	expTable.reserve(exp_table_size + 1);
	expTable.resize(exp_table_size + 1);
	for (auto i = 0; i < exp_table_size; i++)
	{
		expTable[i] = exp((i / static_cast<myreal>(exp_table_size) * 2 - 1) * max_exp); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
	}
}


void Volume2Vector::printParameterInformation()
{
	cout << "alpha:\t" << this->alpha << endl;
	cout << "sample:\t" << this->sample << endl;
	cout << "hs:\t" << this->hs << endl;
	cout << "negative:\t" << this->negative << endl;
	cout << "min_count:\t" << this->min_count << endl;
	cout << "classes:\t" << this->classes << endl;
	cout << "distance:\t" << this->distance2 << endl;
	cout << "histogram_size:\t" << this->histogramDimension << endl;
	cout << "sample_mode:\t" << this->sample_mode << endl;
	cout << "random_sample_number:\t" << this->random_iteration << endl;
	cout << "vector_size:\t" << this->layer1_size << endl;
	cout << "voxelBasedHistogramSaved:\t" << this->isVoxelBasedHistogramSaved << endl;
	cout << "min_count:\t" << this->min_count << endl;
	cout << "is_two_volume_used:\t" << this->is_two_volume_used << endl;
	cout << "is_hierarchical_softmax_used:\t" << this->hs << endl;

	cout << "dimension:\t" << this->dimension.x << " " << this->dimension.y << " "
		<< this->dimension.z << endl;
	cout << "output_mode" << this->output_mode << endl;
	cout << "only out of range negative used:\t" << onlyOutOfRangeNegativeUsed << endl;
	cout << "is_skip_gram_used:\t" << is_skip_gram_used << endl;

	if (is_two_volume_used) is_multi_volume_used = false;

}

/**
 * \brief 
 * \param regular_1D_data : origin data with range from between 0 to histogramDimension
 * \param dimension : the dimension of origin data
 * \param histogramDimension : the histogram dimension
 * \param layer1_size : the vector length of one word in the vocab
 * \param distance2 : the window of the volume. distance=1(surrounding 6 voxel of the center voxel);distance=2(surrounding 18 voxel of the center voxel);distance=3(surrounding 26 voxel of the center voxel)
 * \param negative : Number of negative examples; default is 0, common values are 3 - 10 (0 = not used)????
 * \param random_iteration : Number of random iterations; default is 26, cube of 3*3*3-1. 
 * \param isVoxelBasedHistogramSaved 
 * \param min_count : This will discard words that appear less than <int> times; default is 5
 * \param output_mode : Output word classes rather than word vectors; default number of classes is 0 (vectors are written), =2 is the cluster, =3 is similarity array
 * \param sample : Set threshold for occurrence of words. Those that appear with higher frequency in the training data
 * \param alpha : Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
 * \param sample_mode : The mode of sample. 0 is voxel-based and 1 for the value-based.
 * \param onlyOutOfRangeNegativeUsed
 * \param classes : The number of k-means cluster. default =3.
 */
void Volume2Vector::setInformation(std::vector<int>& regular_1D_data,
	my_int3& dimension, int histogramDimension,
	int layer1_size, int distance2, int negative,
	int random_iteration, int isVoxelBasedHistogramSaved, int min_count,
	int output_mode, myreal sample, myreal alpha, int sample_mode,
	bool onlyOutOfRangeNegativeUsed, int classes, int hs, bool is_skip_gram_used)
{
	regularData.clear();
	regularData.resize(dimension.z);
	for (auto i = 0;i<dimension.z;i++)
	{
		regularData[i].resize(dimension.y);
		for (auto j = 0;j<dimension.y;j++)
		{
			regularData[j].resize(dimension.x);
			for (auto k = 0;k<dimension.x;k++)
			{
				regularData[i][j][k] = regular_1D_data[i*dimension.y*dimension.x + j*dimension.x + k];
			}
		}
	}
	this->dimension = dimension;
	this->histogramDimension = histogramDimension;
	this->layer1_size = layer1_size;
	this->distance2 = distance2;
	this->negative = negative;
	this->random_iteration = random_iteration;
	this->isVoxelBasedHistogramSaved = isVoxelBasedHistogramSaved;
	this->min_count = min_count;
	this->output_mode = output_mode;
	this->sample = sample;
	this->alpha = alpha;
	this->sample_mode = sample_mode;
	this->onlyOutOfRangeNegativeUsed = onlyOutOfRangeNegativeUsed;
	this->classes = classes;
	this->is_two_volume_used = false;
	this->is_multi_volume_used = false;
	this->hs = hs;
	this->is_skip_gram_used = is_skip_gram_used;
	vocab.resize(histogramDimension);

	printParameterInformation();
}


void Volume2Vector::clear()
{
	syn0.clear();
	syn1.clear();
	syn1neg.clear();
	vocab.clear();
	regularData.clear();
	vocab_size = 0;
}

void Volume2Vector::init()
{
	initNet();
	if(negative>0) initUnigramTable();

	for (auto i = 0;i < vocab_size;i++) {
		cout << vocab[i].intword << "\t" << vocab[i].cn << "\t" << vocab_hash[vocab[i].intword] << "\t" << vocab_hash[i] << "\t";
		for (auto j = 0;j < vocab[i].codelen;j++)
		{
			cout << static_cast<int>(vocab[i].code[j]);
		}
		cout << endl;
	}
}
Volume2Vector::~Volume2Vector()
{
	syn0.clear();
	syn1.clear();
	syn1neg.clear();
	vocab.clear();
	table.clear();
	expTable.clear();
	vocab_hash.clear();
}
void Volume2Vector::setRegularVolume(vector<vector<vector<int>>>& regularData, my_int3& dimension)
{
	this->regularData = regularData;
	this->dimension = dimension;
}
int Volume2Vector::getWordHash(const int intword) const
{
	return intword;
}

int Volume2Vector::addWordToVocab(int intword)
{
	vocab[vocab_size].intword = intword;
	vocab[vocab_size].cn = 0;
	vocab_size++;

	const unsigned int hash = getWordHash(intword);

	vocab_hash[hash] = vocab_size - 1;

	return vocab_size - 1;
}
/**
 * \brief Returns position of a word in the vocabulary; if the word is not found, returns -1
 * \param intword 
 * \return 
 */
int Volume2Vector::searchVocab(const int intword) {
	const unsigned int hash = getWordHash(intword);
	return vocab_hash[hash];
}

void Volume2Vector::learnVocab()
{
	if(regularData.empty())
		return;
	int a;
	vocab_hash.clear();
	vocab_hash.resize(vocab_hash_size);
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;

	vocab_size = 0;

	addWordToVocab(0);
	for (auto k = 0; k < dimension.z; k++)
	{
		for (auto j = 0; j < dimension.y; j++)
		{
			for (auto i = 0; i < dimension.x; i++)
			{
				auto& intword = regularData[k][j][i];
				train_words++;
				if ((debug_mode > 1) && (train_words % 10000 == 0)) {
					printf("Loading data to vocab: %f%%%c", train_words*1.0f / (dimension.x*dimension.y*dimension.z) * 100, 13);
					fflush(stdout);
				}
				const auto index = searchVocab(intword);
				if (index <0) {
					a = addWordToVocab(intword);
					vocab[a].cn = 1;
				}
				else vocab[index].cn++;
			}
		}
	}
	sortVocab();

	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", static_cast<long long>(vocab_size));
		printf("Words in train file: %lld\n", train_words);
	}
}

/**
 * \brief Sorts the vocabulary by frequency using word counts
 */
void Volume2Vector::sortVocab() {
	int a;
	sort(vocab.begin(), vocab.begin() + vocab_size);
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	const auto size = vocab_size;
	train_words = 0;
	//
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if ((vocab[a].cn <= min_count) /*&& (a != 0)*/) {
			vocab_size--;
		}
		else {
			// Hash will be re-computed, as after the sorting it is not actual
			const unsigned int hash = getWordHash(vocab[a].intword);
			
			vocab_hash[hash] = a;
			train_words += vocab[a].cn;
		}
	}

}
void Volume2Vector::initUnigramTable()
{
	long long a;
	double train_words_pow = 0;
	const auto power = 0.75;
	table.reserve(table_size);
	table.resize(table_size);
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
	auto i = 0;
	auto d1 = pow(vocab[i].cn, power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / static_cast<double>(table_size) > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1;
	}

}


void Volume2Vector::initNet()
{
	long long a, b;
	unsigned long long next_random = 1;

	syn0.clear();
	syn0.reserve(vocab_size * layer1_size);
	syn0.resize(vocab_size * layer1_size);

	if (hs) {
		syn1.clear();
		syn1.reserve(vocab_size * layer1_size);
		syn1.resize(vocab_size * layer1_size);
		for (a = 0; a < vocab_size; a++) 
			for (b = 0; b < layer1_size; b++)
				syn1[a * layer1_size + b] = 0;
	}

	if (negative>0) {
		syn1neg.clear();
		syn1neg.reserve(vocab_size * layer1_size);
		syn1neg.resize(vocab_size * layer1_size);

		for (a = 0; a < vocab_size; a++)
			for (b = 0; b < layer1_size; b++)
				syn1neg[a * layer1_size + b] = 0;
	}
	for (a = 0; a < vocab_size; a++)
		for (b = 0; b < layer1_size; b++) {
			next_random = next_random * static_cast<unsigned long long>(25214903917) + 11;
			syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / static_cast<myreal>(65536)) - 0.5) / layer1_size;
		}
	createBinaryTree();
}
void Volume2Vector::createBinaryTree() {
	long long a, b, i, min1i, min2i, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];

	std::vector<long long> count;
	count.reserve(vocab_size * 2 + 1);
	count.resize(vocab_size * 2 + 1);

	std::vector<long long> binary;
	binary.reserve(vocab_size * 2 + 1);
	binary.resize(vocab_size * 2 + 1);

	std::vector<long long> parent_node;
	parent_node.reserve(vocab_size * 2 + 1);
	parent_node.resize(vocab_size * 2 + 1);

	//auto* count = static_cast<long long *>(calloc(vocab_size * 2 + 1, sizeof(long long)));
	//auto* binary = static_cast<long long *>(calloc(vocab_size * 2 + 1, sizeof(long long)));
	//auto* parent_node = static_cast<long long *>(calloc(vocab_size * 2 + 1, sizeof(long long)));
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
	long long pos1 = vocab_size - 1;
	long long pos2 = vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (a = 0; a < vocab_size - 1; a++) {
		// First, find two smallest nodes 'min1, min2'
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			}
			else {
				min1i = pos2;
				pos2++;
			}
		}
		else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			}
			else {
				min2i = pos2;
				pos2++;
			}
		}
		else {
			min2i = pos2;
			pos2++;
		}
		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b >= vocab_size * 2 - 2||b<0) break;
		}
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;
		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	//free(count);
	//free(binary);
	//free(parent_node);
}

void Volume2Vector::initLambda()
{
	std::vector<int> edge_table(histogramDimension);
	int edge_number = 0;
	for (auto i = 0;i<histogramDimension;i++)
	{
		edge_table[i] = 0;
		for (auto j = 0;j<histogramDimension;j++)
		{
			if (histogramBasedVoxel[i][j]>0)
				edge_table[i]++;
		}
		edge_number += edge_table[i];
	}

	lambda_array.clear();
	lambda_array.resize(vocab_size);
	const auto ll = log2(layer1_size) / 6 + 1.9;
	const myreal n = vocab_size;
	const auto k = negative;
	for (auto i = 0;i<vocab_size;i++)
	{
		//const auto m = log(1 + edge_table[vocab[i].intword] * 1.0 / n)/log(1+n);
		const auto m = log(1 + edge_number * 1.0 / n) / log(1 + n);
		const auto first_item = m / (1 + exp(ll*ll*m));
		const auto second_item = k*(sqrt(1 - m*m)) / (1 + exp(ll*ll*sqrt(1 - m*m)));
		lambda_array[i] = (first_item + second_item) / 2;
	}
}

void Volume2Vector::trainMultiThreading()
{
	long long id = 0;
	long long a, b, d, cw, word_index = 0, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0;

	long long l1, l2, c, target, label;
	unsigned long long next_random = (long long)id;
	myreal f, g;
	//clock_t now;
	long long xDim = dimension.x;
	long long yDim = dimension.y;
	long long zDim = dimension.z;
	long long sentance_size = xDim*yDim*zDim;

	std::vector<int> sentance;
	sentance.reserve(sentance_size);
	sentance.resize(sentance_size);

	int intword;


	histogramBasedVoxel.clear();
	histogramBasedVoxel.resize(histogramDimension);
	for (auto i = 0;i<histogramDimension;i++)
	{
		histogramBasedVoxel[i].reserve(histogramDimension);
		histogramBasedVoxel[i].resize(histogramDimension);
	}


	neighborhood_distribution_array.clear();
	neighborhood_distribution_array.resize(sentance_size);
	
	auto time_begin = clock();

	std::cout << "There are " << omp_get_num_procs() << " cores in this PC." << std::endl;  //Gets the current number of computer processors
	int num_cores;
	num_cores = omp_get_num_procs() * 3 / 4;
	omp_set_num_threads(num_cores);   //Specifies the number of threads used for parallel computation

#ifdef IS_MULTITHREADING_USED
#pragma omp parallel for
#endif
	for (long long index = 0; index < sentance_size; index++) {

		//initialize the temp value array;
		initializeNeighborhoodDistribution(index);


	}
	/*#ifdef IS_MULTITHREADING_USED
	#pragma omp barrier
	#endif*/
	auto time_end = clock();
	std::cout << "Time for initializing neighborhood distribution : " << (time_end - time_begin)*1.0f / 1000.0f << std::endl;
	std::cout << std::endl;


	if (negative>0)
	{
		initLambda();
	}

	//Debug 20190505
	//Initialize the negative sample array
	negative_sample_array.clear();
	negative_sample_array.resize(histogramDimension);

	negative_probability.clear();
	negative_probability.resize(histogramDimension);

	for (auto i = 0;i<histogramDimension;i++)
	{
		negative_probability[i].resize(histogramDimension);
		auto& cur_array = histogramBasedVoxel[i];
		for (auto j = 0;j < cur_array.size();j++)
		{
			if (cur_array[j] == 0 && i != j/*&&vocab_hash[j]>=0*/) negative_sample_array[i].push_back(j);//ֵ
		}
	}


	auto starting_alpha = alpha;

	//#pragma omp parallel for
	//long long word_count, last_word_count;
	long long  word_count_actual = 0;
	int cnt = 0;
	int max_number = sentance_size / 10000 - 1;
	myreal bb = 916.421237604 / max_number;

	pram_a = 4 * 1.0 / (max_number*max_number*1.0);
	pram_b = 1e-5;
	u = 1.0;
	std::cout << max_number << "\t" << bb << "; a:" << pram_a << "; b:" << pram_b << std::endl;

	std::vector<int> train_times_array(256);
	int is_debug = false;



	for (long long index = 0; index < sentance_size; index++) {
		if (index % 10000 == 0) {
			//For manix dataset, 0.9935 is available.
			alpha = starting_alpha * pow(0.9945, bb*(cnt++));
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
			u++;
			Lu = std::min(pram_a*u*u + pram_b, 1.0);
			//Lu = 1;

			printf("u: %llf; Lu: %llf; count: %d; word_count_actual: %lld; Alpha: %f; Degree of completion: %f%%%c", u, Lu,
				cnt, word_count_actual, alpha, (index*1.0f / sentance_size * 100), 13);
			fflush(stdout);
		}

		if (neighborhood_distribution_array[index].empty()) continue;
		
		skipGramsMultiTheading(index, sentance_size);
	}
		
	
	std::cout << "Vocab size: " << vocab_size << std::endl;
	std::cout << "Words in train file: " << train_words << std::endl;
	std::cout << std::endl;
}

void Volume2Vector::initializeNeighborhoodDistribution(long long index)
{
	long long word_index = 0;

	auto currentIndex = 0;
	const auto offset = static_cast<int>(sqrt(distance2));
	const auto& xDim = dimension.x;
	const auto& yDim = dimension.y;
	const auto& zDim = dimension.z;

	neighborhood_distribution_array[index].clear();

	const int k = index / (xDim*yDim);
	const int i = index%xDim;
	const int j = (index % (xDim*yDim)) / xDim;

	const auto& centerValue = regularData[k][j][i];
	static long long next_random = rand() % 65536;

	if (centerValue<0 || centerValue>histogramDimension) return;

	word_index = vocab_hash[centerValue];
	if (word_index == -1 || word_index >= histogramDimension) return;
	
	vector<int> buf_array;
	for (auto zvalue = k - offset;zvalue <= k + offset;zvalue++) {
		if (zvalue<0 || zvalue >= regularData.size()) continue;
		//for y
		for (auto yvalue = j - offset;yvalue <= j + offset;yvalue++) {
			if (yvalue<0 || yvalue >= regularData[0].size()) continue;
			//for x
			for (auto xvalue = i - offset;xvalue <= i + offset;xvalue++) {
				if (xvalue<0 || xvalue >= regularData[0][0].size()) continue;

				if (abs(zvalue - k) + abs(yvalue - j) + abs(xvalue - i)>distance2) continue;
				if (zvalue == k&&yvalue == j&&xvalue == i) continue;
				currentIndex = zvalue*xDim*yDim + yvalue*xDim + xvalue;

				if (currentIndex>xDim*yDim*zDim - 1 || currentIndex<0) continue;

				auto& int_word = regularData[zvalue][yvalue][xvalue];
				if (int_word<0 || int_word>histogramDimension) continue;
				if (vocab_hash[int_word] <0) continue;

				//TODO::
				if (isVoxelBasedHistogramSaved)
				{
#pragma omp atomic
					histogramBasedVoxel[centerValue][int_word]++;
				}
				buf_array.push_back(int_word);
			}
		}
	}

	// The subsampling randomly discards frequent words while keeping the ranking same
	if (sample > 0) {
		{
			myreal ran = (sqrt(vocab[word_index].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word_index].cn;
			next_random = next_random * (unsigned long long)25214903917 + 11;
			if (ran < (next_random & 0xFFFF) / (myreal)65536) {
				return;
			}
		}
	}

	neighborhood_distribution_array[index] = buf_array;
	auto& tempValueArray = neighborhood_distribution_array[index];
	if (sample_mode) {

		if (this->is_two_volume_used&&tempValueArray.size()>0)
		{
			const auto single_dimension = static_cast<int>(sqrt(histogramDimension));

			int max_fixed_value = -1, min_fixed_value = histogramDimension * 1000;
			int max_test_value = -1, min_test_value = histogramDimension * 1000;

			for (auto& temp_value : tempValueArray)
			{
				int fixed_value = temp_value / single_dimension;
				int test_value = temp_value % single_dimension;
				
				max_fixed_value = std::max(max_fixed_value, fixed_value);
				min_fixed_value = std::min(min_fixed_value, fixed_value);

				max_test_value = std::max(max_test_value, test_value);
				min_test_value = std::min(min_test_value, test_value);
			}
			tempValueArray.clear();
			for (auto d = 0;d < random_iteration;d++) {
				const int rand_fixed_value = rand() % (max_fixed_value - min_fixed_value + 1) + min_fixed_value;
				const int rand_test_value = rand() % (max_test_value - min_test_value + 1) + min_test_value;

				int rand_composed_value = rand_fixed_value*single_dimension + rand_test_value;

				if (vocab_hash[rand_composed_value] <0 || vocab_hash[rand_composed_value] >= histogramDimension) continue;

				tempValueArray.push_back(rand_composed_value);
			}
		}
		else if(is_multi_volume_used&&tempValueArray.size() > 0)
		{
			const int single_dimension = this->single_dimension;

			std::vector<int> max_value_array(multivariate_number);
			std::vector<int> min_value_array(multivariate_number);
			
			for (auto i = 0; i < multivariate_number; i++)
			{
				max_value_array[i] = -1;
				min_value_array[i] = histogramDimension * 1000;
			}

			for (auto temp_value : tempValueArray)
			{
				std::vector<int> current_value_array(multivariate_number);
				for (auto i = multivariate_number - 1; i >= 0; i--)
				{
					current_value_array[i] = temp_value % single_dimension;
					max_value_array[i] = std::max(current_value_array[i], max_value_array[i]);
					min_value_array[i] = std::min(current_value_array[i], min_value_array[i]);
					temp_value /= single_dimension;
				}
			}
			tempValueArray.clear();

			for (auto d = 0; d < random_iteration; d++) {
				int rand_composed_value = 0;
				for(auto i=0;i<multivariate_number;i++)
				{
					const auto current_value = rand() % (max_value_array[i] - min_value_array[i] + 1)+min_value_array[i];
					rand_composed_value += pow(single_dimension, multivariate_number - 1 - i) * current_value;
				}
				if (vocab_hash[rand_composed_value] < 0 || vocab_hash[rand_composed_value] >= histogramDimension) continue;
				tempValueArray.push_back(rand_composed_value);
			}
		}
		else
		{
			auto maxValue = -1;
			auto minValue = histogramDimension * 100;
			
			for (auto& temp_value : tempValueArray)
			{
				maxValue = std::max(maxValue, temp_value);
				minValue = std::min(minValue, temp_value);
			}
			tempValueArray.clear();
			for (auto d = 0;d < random_iteration;d++) {
				auto randValue = rand() % (maxValue - minValue + 1) + minValue;

				if (vocab_hash[randValue] <0 || vocab_hash[randValue] >= histogramDimension) continue;
				tempValueArray.push_back(randValue);
			}
		}
	}
	else {

	}

}

void Volume2Vector::skipGramsMultiTheading(long long index, long long sentance_size)
{

	unsigned long long next_random = rand() % 65536;
	int c, d;


	const int k = index / (dimension.x*dimension.y);
	const int i = index%dimension.x;
	const int j = (index % (dimension.x*dimension.y)) / dimension.x;


	const auto& centerValue = regularData[k][j][i];
	auto& cur_negative_sample_array = negative_sample_array[centerValue];

	
	if (centerValue<0 || centerValue>histogramDimension) return;
	auto word_index = vocab_hash[centerValue];


	if (word_index == -1 || word_index >= histogramDimension) return;


	std::vector<myreal> neu1e;
	neu1e.reserve(layer1_size);
	neu1e.resize(layer1_size);


	auto cw = 0;

	for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

	auto maxValue = -1;
	auto minValue = histogramDimension * 100;


	auto random_sample_value = 0;

	auto& tempValueArray = neighborhood_distribution_array[index];

	myreal loss_function_for_voxel = 0.0;

	for (d = 0;d < tempValueArray.size();d++) {
		if (vocab_hash[tempValueArray[d]] <0 || vocab_hash[tempValueArray[d]] >= histogramDimension) continue;

		auto l1 = vocab_hash[tempValueArray[d]] * layer1_size;

		for (c = 0;c < layer1_size;c++) neu1e[c] = 0;
		//Negative sampling
		if (negative>0)
		{
			long long target = 0;
			int label = 0;

			std::set<int> negative_word_set;
			std::vector<int> negative_filter_array;
			std::vector<myreal> fu_array;
			myreal fu_norm = 0;

			
			auto l0 = word_index*layer1_size;
			fu_array.resize(layer1_size);
			for (c = 0;c<layer1_size;c++)
			{
				fu_norm += syn0[c + l1] * syn0[c + l1];
			}
			fu_norm = sqrt(fu_norm);
			if (fu_norm<1e-8)
			{
				for (c = 0;c<layer1_size;c++)
					fu_array[c] = syn0[c + l1];
			}
			else
			{
				for (c = 0;c<layer1_size;c++)
					fu_array[c] = syn0[c + l1] / fu_norm;
			}
			if (is_self_paced_embedding_used)
			{
				for (c = 0;c<cur_negative_sample_array.size();c++)
				{
					if (negative_probability[word_index][vocab_hash[cur_negative_sample_array[c]]]<Lu)
					{
						negative_filter_array.push_back(cur_negative_sample_array[c]);
					}
				}
			}

			for (int i = 0;i<negative + 1;i++)
			{
				if (i == 0)
				{
					target = word_index;
					label = 1;
					//L2 is the node of negative samples. When i=0, l2 is the node of context(u)
					long long l2 = target*layer1_size;
					//l2 = l1;
					myreal f = 0.0;
					myreal g = 0.0;
					for (c = 0;c < layer1_size;c++) f += syn0[c + l1] * syn1neg[c + l2];

					if (f > max_exp) g = (label - 1) * alpha;
					else if (f < -max_exp) g = (label - 0) * alpha;
					else g = (label - expTable[static_cast<int>((f + max_exp) * (exp_table_size / max_exp / 2))])/* * alpha*/;

					for (c = 0;c < layer1_size;c++) neu1e[c] += (g*syn1neg[c + l2] - lambda_array[target] * fu_array[c])*alpha;
					for (c = 0;c < layer1_size;c++) syn1neg[c + l2] += g*syn0[c + l1] * alpha;
				}
				//Negative sampling does not sample the non-adjacent points corresponding to the context(w). It samples the non-adjacent points corresponding to w
				else
				{
					next_random = next_random * static_cast<unsigned long long>(25214903917) + 11;
					int negative_sample_size;
					int word;
					if (is_self_paced_embedding_used)
					{
						negative_sample_size = negative_filter_array.size();
						if (negative_sample_size == 0) continue;
						else word = negative_filter_array[next_random % (negative_sample_size)];
					}
					else
					{
						negative_sample_size = cur_negative_sample_array.size();
						if (negative_sample_size == 0) /*word = next_random % histogramDimension;*/continue;
						else word = cur_negative_sample_array[next_random % (negative_sample_size)];
					}
					target = vocab_hash[word];
					if (target<0 || target>histogramDimension) continue;
					negative_word_set.insert(word);
					label = 0;

					long long l2 = target*layer1_size;
					myreal f = 0.0;
					myreal g = 0.0;
					for (c = 0;c < layer1_size;c++) f += syn0[c + l1] * syn1neg[c + l2];

					if (f > max_exp) g = (label - 1) * alpha;
					else if (f < -max_exp) g = (label - 0) * alpha;
					else g = (label - expTable[static_cast<int>((f + max_exp) * (exp_table_size / max_exp / 2))])/* * alpha*/;

					std::vector<myreal> fvj_array(layer1_size);
					myreal fvj_norm = 0;
					for (c = 0;c < layer1_size;c++) fvj_norm += syn1neg[c + l2] * syn1neg[c + l2];
					fvj_norm = sqrt(fvj_norm);
					if (fvj_norm<1e-8)
					{
						for (c = 0;c < layer1_size;c++) fvj_array[c] = syn1neg[c + l2];
					}
					else
					{
						for (c = 0;c < layer1_size;c++) fvj_array[c] = syn1neg[c + l2] / fvj_norm;
					}
					for (c = 0;c < layer1_size;c++) neu1e[c] += g*syn1neg[c + l2] * alpha;
					//Debug 20190510 l1->l0
					for (c = 0;c < layer1_size;c++) syn1neg[c + l2] += (g*syn0[c + l1] - lambda_array[target] / (negative + 1)*fvj_array[c])*alpha;
				}
			}

			if (is_self_paced_embedding_used)
			{
				//Update the sampling probability pij as Eq.5 and Eq.6
				map<int, double> f_map;
				myreal sum_f = 0.0;
				for (auto p : negative_word_set)
				{
					const auto buf_target = vocab_hash[p];
					const long long l2 = buf_target * layer1_size;
					myreal f = 0;
					
					//for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
					for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn0[c + l2];
					sum_f += exp(f);
					f_map[p] = exp(f);
				}
				if (sum_f>1e-8)
				{
					for (auto p : negative_word_set) negative_probability[centerValue][p] = f_map[p] / sum_f;
				}
				else
				{
					for (auto p : negative_word_set) negative_probability[centerValue][p] = 0.0;
				}
			}
			//Learning weights input -> hidden
			for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];


			myreal buf_loss_function_for_voxel = 0.0;
			for (c = 0; c < layer1_size; c++)
			{
				buf_loss_function_for_voxel += syn0[c + l0] * syn1neg[c + l1];
			}
			//if (buf_loss_function_for_voxel > max_exp) buf_loss_function_for_voxel = 0;
			//else if (buf_loss_function_for_voxel < -max_exp) buf_loss_function_for_voxel = -1;
			//else buf_loss_function_for_voxel = log10(expTable[static_cast<int>((buf_loss_function_for_voxel + max_exp) * (exp_table_size / max_exp / 2))]);
			buf_loss_function_for_voxel = log(1.0 / (exp(-buf_loss_function_for_voxel) + 1));
			loss_function_for_voxel += buf_loss_function_for_voxel;
		}
	}

}

/**
* \brief Calculate the similarity between index_a and index_b. If the cos value < 0, we need set it to 0
* \param index_a
* \param index_b
* \return
*/
myreal Volume2Vector::similarity(const int index_a, const int index_b) {


	if (index_a <0 || index_b <0) return 0.0f;
	int b;

	vector<myreal> center;

	center.reserve(layer1_size);
	center.resize(layer1_size);
	for (b = 0; b < layer1_size; b++)
		center[b] = syn0[index_a * layer1_size + b];

	vector<myreal> compare;
	compare.reserve(layer1_size);
	compare.resize(layer1_size);
	for (b = 0; b < layer1_size; b++)
		compare[b] = syn0[index_b * layer1_size + b];

	myreal dist = 0;
	myreal m_a = 0, m_b = 0;
	for (int i = 0; i < layer1_size; i++) {
		dist += center[i] * compare[i];
		m_a += center[i] * center[i];
		m_b += compare[i] * compare[i];
	}

	m_a = sqrt(m_a);
	m_b = sqrt(m_b);
	if (dist < 0) return 0.0f;

	if (m_a <= 1e-8 || m_b <= 1e-8) return 0.0f;
	else
		dist = dist / (m_a*m_b);
	return dist;
}

std::vector<std::vector<myreal>>& Volume2Vector::getFeatureVector()
{
	feature_vector.clear();
	feature_vector.resize(histogramDimension);

	for (auto& i : feature_vector)
	{
		i.reserve(layer1_size);
		i.resize(layer1_size);

		for (auto& j : i)
		{
			j = 0.0;
		}
	}
	for(auto i=0;i<vocab_size;i++)
	{
		auto& word_value = vocab[i].intword;
		if (word_value<0 || word_value>histogramDimension) continue;
		for (auto j = 0;j<layer1_size;j++)
		{
			feature_vector[word_value][j] = syn0[i*layer1_size + j];
		}
	}
	
	std::cout << "Feature vector has been translated." << std::endl;
	return feature_vector;
}

void Volume2Vector::setFileSaveState(bool is_save_histogram_distribution, bool is_save_similarity_map,
	bool is_save_volume_vector, bool is_save_cluster)
{
	this->is_save_histogram_distribution = is_save_histogram_distribution;
	this->is_save_similarity_map = is_save_similarity_map;
	this->is_save_volume_vector = is_save_volume_vector;
	this->is_save_cluster = is_save_cluster;
}
