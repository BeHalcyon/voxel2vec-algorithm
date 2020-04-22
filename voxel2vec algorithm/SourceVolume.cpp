#include "SourceVolume.h"
#include <iostream>
#include <fstream>


/**
 * \brief A template function to get the origin volume data(target_data).
 */
template <class VolumeType>
void loadAndTransVolume(std::vector<std::string>& volume_file_name, hxy::my_int3& volume_res, std::vector<hxy::myreal>& min_value, std::vector<hxy::myreal>& max_value,
	std::vector<std::vector<VolumeType>>& origin_data, std::vector<std::vector<hxy::myreal>>& target_data)
{
	origin_data.clear();
	origin_data.resize(volume_file_name.size());
	target_data.clear();
	target_data.resize(volume_file_name.size());
	for (auto i = 0; i < volume_file_name.size(); i++)
	{
		std::ifstream in(volume_file_name[i].toStdString(), std::ios::in | std::ios::binary);
		unsigned char *contents = nullptr;
		if (in)
		{
			in.seekg(0, std::ios::end);
			const long int fileSize = in.tellg();
			contents = static_cast<unsigned char*>(malloc(static_cast<size_t>(fileSize + 1)));
			in.seekg(0, std::ios::beg);
			in.read(reinterpret_cast<char*>(contents), fileSize);
			in.close();
			contents[fileSize] = '\0';
			std::cout << "Load data successfully.\nThe file path is : " << volume_file_name[i].toStdString() << std::endl;

			std::cout << "The file size is : " << fileSize << std::endl;

			
			const long long volume_length = volume_res.z*volume_res.y*volume_res.x;
			origin_data[i].resize(volume_length);
			target_data[i].resize(volume_length);
			for (auto x = 0; x < volume_length; ++x)
			{
				int src_idx = sizeof(VolumeType) * (x);
				memcpy(&origin_data[i][x], &contents[src_idx], sizeof(VolumeType));
				max_value[i] = origin_data[i][x]> max_value[i] ? origin_data[i][x] : max_value[i];
				min_value[i] = origin_data[i][x]< min_value[i] ? origin_data[i][x] : min_value[i];

				target_data[i][x] = origin_data[i][x];
			}
			std::cout << "Max value : " << max_value[i] << " min_value : " << min_value[i] <<std::endl;
		}
		else
		{
			std::cout << "The file " << volume_file_name[i].toStdString() << " fails loaded." << std::endl;
		}
		free(contents);
	}
}


SourceVolume::SourceVolume(std::string type_name, int regular_histogram_dimension)
	: type_name(type_name), regular_histogram_dimension(regular_histogram_dimension), volume_number(0),
	  volume_length(0),
	  volume_res(0, 0, 0), fixed_index(0), test_index(0)
{
	is_data_in_memory = false;
	is_regular_data_generated = false;
	is_down_sampling_data_generated = false;
	is_histogram_distribution_calculated = false;
}

SourceVolume::SourceVolume(std::vector<std::string> file_name, int x, int y, int z, std::string type_name, int regular_histogram_dimension, int downing_sampling_histogram_dimension)
	:type_name(type_name), regular_histogram_dimension(regular_histogram_dimension), volume_length(0), 
	fixed_index(0), test_index(0), downing_sampling_histogram_dimension(downing_sampling_histogram_dimension)
{
	is_data_in_memory = false;
	is_regular_data_generated = false;
	is_down_sampling_data_generated = false;
	is_histogram_distribution_calculated = false;

	setFileAndRes(file_name, x, y, z);
}

SourceVolume::SourceVolume()
{
}

void SourceVolume::loadVolume()
{
	if (is_data_in_memory)
	{
		std::cout << "The origin data has already been loaded." << std::endl;
		return;
	}
	min_value = std::vector<hxy::myreal>(volume_number, 1e8);
	max_value = std::vector<hxy::myreal>(volume_number, -1e8);

	if(type_name=="uchar")
	{
		std::vector<std::vector<unsigned char>> origin_data;
		loadAndTransVolume(volume_file_name, volume_res, min_value, max_value, origin_data, volume_data);
	}
	else if(type_name=="ushort")
	{
		std::vector<std::vector<unsigned short>> origin_data;
		loadAndTransVolume(volume_file_name, volume_res, min_value, max_value, origin_data, volume_data);
	}
	else if(type_name=="float")
	{
		std::vector<std::vector<float>> origin_data;
		loadAndTransVolume(volume_file_name, volume_res, min_value, max_value, origin_data, volume_data);
	}
	is_data_in_memory = true;
	std::cout << "The origin volume data has been calculated successfully." << std::endl;
}

void SourceVolume::loadRegularVolume()
{
	if(!is_data_in_memory)
	{
		loadVolume();
	}
	if(is_regular_data_generated)
	{
		std::cout << "The regular data has already been loaded." << std::endl;
		return;
	}
	//The next step under the condition that the origin data has been loaded to memory.
	regular_data.clear();
	regular_data.resize(volume_number);
	//const long long volume_length = volume_res.z*volume_res.y*volume_res.x;
	for(auto i=0;i<volume_number;i++)
	{
		regular_data[i].resize(volume_length);
		for(auto index = 0;index<volume_length;index++)
		{
			regular_data[i][index] = (volume_data[i][index] - min_value[i]) / (max_value[i] - min_value[i])*(regular_histogram_dimension - 1);
		}
	}
	is_regular_data_generated = true;
	std::cout << "The regular volume data has been calculated successfully." << std::endl;
}

void SourceVolume::loadDownSamplingVolume()
{
	if (!is_data_in_memory)
	{
		loadVolume();
	}
	if (is_down_sampling_data_generated)
	{
		std::cout << "The down sampling data has already been loaded." << std::endl;
		return;
	}

	//The next step under the condition that the origin data has been loaded to memory.
	down_sampling_data.clear();
	down_sampling_data.resize(volume_number);
	down_sampling_uchar_data.clear();
	down_sampling_uchar_data.resize(volume_number);
	for (auto i = 0;i<volume_number;i++)
	{
		down_sampling_data[i].resize(volume_length);
		down_sampling_uchar_data[i].resize(volume_length);
		for (auto index = 0;index<volume_length;index++)
		{
			down_sampling_data[i][index] = (volume_data[i][index] - min_value[i]) / (max_value[i] - min_value[i])*(downing_sampling_histogram_dimension - 1);
			down_sampling_uchar_data[i][index] = down_sampling_data[i][index];
		}
	}
	is_down_sampling_data_generated = true;
	std::cout << "The down sampling data has been calculated successfully." << std::endl;
}

std::vector<unsigned char>* SourceVolume::getRegularVolume(int index)
{
	if(index<0||index>=volume_number)
	{
		std::cout << "The index out of bounds in getRegularVolume" << std::endl;
		exit(-1);
	}
	if (!is_regular_data_generated)
	{
		loadRegularVolume();
	}
	return &(regular_data[index]);
}

std::vector<int>* SourceVolume::getDownsamplingVolume(int index)
{
	if (index<0 || index >= volume_number)
	{
		std::cout << "The index out of bounds." << std::endl;
		exit(-1);
	}
	if (!is_down_sampling_data_generated)
	{
		loadDownSamplingVolume();
	}
	return &(down_sampling_data[index]);
}
std::vector<unsigned char>* SourceVolume::getDownsamplingUcharVolume(int index)
{
	if (index<0 || index >= volume_number)
	{
		std::cout << "The index out of bounds." << std::endl;
		exit(-1);
	}
	if (!is_down_sampling_data_generated)
	{
		loadDownSamplingVolume();
	}
	return &(down_sampling_uchar_data[index]);
}
/**
 * \brief Get the down sampling volume \note Only two variables are supposed.
 * \param fixed_index 
 * \param test_index 
 * \return 
 */
std::vector<int>* SourceVolume::getDownsamplingVolume(int fixed_index, int test_index)
{
	if (fixed_index<0 || fixed_index >= volume_number || test_index<0 || test_index>volume_number)
	{
		std::cout << "The index out of bounds." << std::endl;
		exit(-1);
	}
	this->fixed_index = fixed_index;
	this->test_index = test_index;
	if (!is_down_sampling_data_generated)
	{
		loadDownSamplingVolume();
	}
	combination_down_sampling_data.clear();
	combination_down_sampling_data.resize(volume_length);
	for(auto i=0;i<volume_length;i++)
	{
		//Calculate the result data;
		const int fixed_value = down_sampling_data[fixed_index][i];
		const int test_value = down_sampling_data[test_index][i];
		combination_down_sampling_data[i] = fixed_value*downing_sampling_histogram_dimension + test_value;
	}
	std::cout << "The down sampling data for index combination [" << fixed_index << ", " << test_index << "] has been calculated." << std::endl;
	return &combination_down_sampling_data;
}

/**
 * \brief Get the down sampling volume \note Multivariate variables are supposed.
 * \param index_array 
 * \return 
 */
std::vector<int>* SourceVolume::getDownsamplingVolume(const std::vector<int>& index_array)
{

	for (int i : index_array)
	{
		if(i <0|| i >=volume_number)
		{
			std::cout << "The index out of bounds." << std::endl;
			exit(-1);
		}
	}

	if (!is_down_sampling_data_generated)
	{
		loadDownSamplingVolume();
	}
	combination_down_sampling_data = std::vector<int>(volume_length, 0);
	for (auto i = 0;i<volume_length;i++)
	{
		for (auto index : index_array)
		{
			combination_down_sampling_data[i] += pow(downing_sampling_histogram_dimension, index_array.size() - index - 1)
												*down_sampling_data[index][i];
		}
	}
	std::cout << "The down sampling data for multivariate data has been calculated." << std::endl;
	return &combination_down_sampling_data;
}


long long SourceVolume::getVolumeSize() const
{
	return volume_length;
}

int SourceVolume::getVolumeNumber() const
{
	return volume_file_name.size();
}


void SourceVolume::setFileAndRes(std::vector<std::string>& file_name, int x, int y, int z)
{
	volume_file_name = file_name;
	volume_number = file_name.size();

	volume_res.x = x;
	volume_res.y = y;
	volume_res.z = z;

	volume_length = volume_res.z*volume_res.y*volume_res.x;
}

void SourceVolume::setFileAndRes(std::vector<std::string>& file_name, hxy::my_int3& resolution)
{
	volume_file_name = file_name;
	volume_number = file_name.size();

	volume_res = resolution;
	volume_length = volume_res.z*volume_res.y*volume_res.x;
}


void SourceVolume::deleteData()
{
	if (is_data_in_memory)
	{
		volume_data.clear();
		is_data_in_memory = false;
	}
	if (is_regular_data_generated)
	{
		regular_data.clear();
		is_regular_data_generated = false;
	}
	if (is_down_sampling_data_generated)
	{
		down_sampling_data.clear();
		is_down_sampling_data_generated = false;
	}
	
	max_value.clear();
	min_value.clear();
	combination_down_sampling_data.clear();
}

SourceVolume::~SourceVolume()
{
}
