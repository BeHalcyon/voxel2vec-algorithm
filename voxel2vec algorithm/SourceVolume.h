
//
//  @ Project : 
//  @ File Name : Volume.h
//  @ Date : 2019/5/20
//  @ Author : 
//
//


#if !defined(_VOLUME_H)
#define _VOLUME_H
#include <vector>
#include "resource.h"
#include <string>

class SourceVolume
{
public:
	

	/**
	 * \brief After this constructor, user don't need to call setFileAndRes function continuously.
	 */
	SourceVolume(std::vector<std::string> file_name, int x, int y, int z, std::string type_name, int histogram_dimension = 256, int downing_sampling_histogram_dimension = 64);
	SourceVolume();
	~SourceVolume();
	
	/**
	 * \brief Load the origin data to memory.
	 * \note The volume data (volume_data) will be loaded at first
	 */
	void loadVolume();

	/**
	* \brief Load the regular data to memory.
	* \note The volume data (regular_data) will be loaded at first
	*/
	void loadRegularVolume();

	/**
	* \brief Load the down sampling data to memory.
	* \note The volume data (down_sampling_data) will be loaded at first
	*/
	void loadDownSamplingVolume();


	/**
	 * \brief Calculate the regularization histogram distribution and origin histogram distribution for multivariate data;
	 */
	void calcHistogramDistribution();

	/**
	 * \brief Get the regular volume data based the index of the volume.
	 * \param index: The volume index.
	 */
	std::vector<unsigned char>* getRegularVolume(int index);
	/**
	 * \brief Get the down sampling data for univariate data.
	 */
	std::vector<int>* getDownsamplingVolume(int index);
	std::vector<unsigned char>* getDownsamplingUcharVolume(int index);
	std::vector<int>* getDownsamplingVolume(int fixed_index, int test_index);
	std::vector<int>* getDownsamplingVolume(const std::vector<int>& index_array);

	/**
	 * \brief 
	 * \note called after load volume to memory.
	 */
	long long getVolumeSize() const;
	int getVolumeNumber() const;


	void deleteData();

	bool inMemory() const { return is_data_in_memory; }
private:

	/**
	* \brief After this constructor, the user must call setFileAndRes function continuously.
	*/
	SourceVolume(std::string type_name, int histogram_dimension = 256);

	void setFileAndRes(std::vector<std::string>& file_name, int x, int y, int z);
	void setFileAndRes(std::vector<std::string>& fileName, hxy::my_int3& resolution);
	


	std::vector<std::string> volume_file_name;

	std::vector<std::vector<hxy::myreal>> volume_data; //the origin data
	std::vector<std::vector<unsigned char>> regular_data; //the max value of each variable is 255
	std::vector<std::vector<int>> down_sampling_data;
	std::vector<std::vector<unsigned char>> down_sampling_uchar_data; //the max value of each variable is 255
	std::vector<int> combination_down_sampling_data;

	//the max value of each variable is downing_sampling_histogram_dimension - 1;
	std::vector<hxy::myreal> max_value, min_value;

	std::string type_name;
	int regular_histogram_dimension = 256;
	int downing_sampling_histogram_dimension = 64;
	bool is_data_in_memory;
	bool is_regular_data_generated;
	bool is_down_sampling_data_generated;

	int volume_number;
	int volume_length;
	hxy::my_int3 volume_res;

	int fixed_index, test_index;
};

#endif  //_VOLUME_H
