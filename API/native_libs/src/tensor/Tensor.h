#ifndef TFL_TENSOR_H
#define TFL_TENSOR_H

#include <vector>
#include <tensorflow/c/c_api.h>
#include <cstring>
#include <cstddef>

#include "API/native_libs/src/utils.h"
#include "TypeLabel.h"

template<TF_DataType DataTypeLabel>
class Tensor {
private:
	TF_Tensor *underlying;
	using type = typename Type<DataTypeLabel>::type;
	std::vector<int64_t> dims;

public:
	explicit Tensor(const type* vect, int64_t len);
	explicit Tensor(const type** array, int64_t width, int64_t height);

	explicit Tensor(const std::vector<type> &vect);
	explicit Tensor(const std::vector<std::vector<type>> &array);

	explicit Tensor(TF_Tensor* underlying) : underlying(underlying) {}

	explicit Tensor(const Tensor& other);

	Tensor(Tensor&& other) noexcept;

	type& at(int64_t const* indices, int64_t len);
	type& at(const std::vector<int64_t> &indices);

	std::vector<int64_t> shape();

	TF_Tensor* get_underlying() const;

	size_t hash() const;

	~Tensor();
};

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor::type *vect, int64_t len)
{
	size_t data_size = TF_DataTypeSize(DataTypeLabel);
	dims = std::vector<int64_t>{len};
	underlying = TF_AllocateTensor(DataTypeLabel, dims.data(), 1, data_size * len);

	auto* adr = (std::byte*) TF_TensorData(underlying);
	for(auto i=0; i<len; ++i)
	{
		*((type*) adr) = vect[i];
		adr += data_size;
	}
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor::type **array, int64_t width, int64_t height)
{
	size_t data_size = TF_DataTypeSize(DataTypeLabel);
	dims = std::vector<int64_t>{width, height};
	underlying = TF_AllocateTensor(DataTypeLabel, dims.data(), 2, width * height * data_size);

	auto* adr = (std::byte*) TF_TensorData(underlying);
	for(auto i=0; i<width; ++i)
	{
		for(auto j=0; j<height; ++j)
		{
			*((type*) adr) = array[j][i];
			adr += data_size;
		}
	}
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const std::vector<Tensor::type> &vect) : Tensor(vect.data(), vect.size()) {}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const std::vector<std::vector<Tensor::type>> &array)
{
	size_t data_size = TF_DataTypeSize(DataTypeLabel);
	dims = std::vector<int64_t>{array.size(), array.front().size()};
	underlying = TF_AllocateTensor(DataTypeLabel, dims.data(), 2, array.size() * array.front().size() * data_size);

	auto* adr = (std::byte*) TF_TensorData(underlying);
	for(auto i=0; i<array.size(); ++i)
	{
		for(auto j=0; j<array[i].size(); ++j)
		{
			*((type*) adr) = array[i][j];
			adr += data_size;
		}
	}
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor &other)
{
	// TODO DEBUG THIS, it likely segfaults
	TF_Tensor *other_underlying = other.get_underlying();
	std::vector<int64_t> other_dims(TF_NumDims(other_underlying));
	auto data_size = TF_TensorByteSize(other_underlying);
	for (size_t i = 0; i < other_dims.size(); ++i) {
		other_dims[i] = TF_Dim(other_underlying, i);
	}
	underlying = TF_AllocateTensor(TF_TensorType(other_underlying), other_dims.data(), other_dims.size(), data_size);
	memcpy(TF_TensorData(underlying), TF_TensorData(other_underlying), data_size);
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(Tensor &&other) noexcept
{
	underlying = other.underlying;
	other.underlying = nullptr;
}

template<TF_DataType DataTypeLabel>
typename Type<DataTypeLabel>::type& Tensor<DataTypeLabel>::at(const std::vector<int64_t> &indices)
{
	return at(indices.data(), indices.size());
}

template<TF_DataType DataTypeLabel>
std::vector<int64_t> Tensor<DataTypeLabel>::shape()
{
	return dims;
}

template<TF_DataType DataTypeLabel>
TF_Tensor *Tensor<DataTypeLabel>::get_underlying() const
{
	return underlying;
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::~Tensor()
{
	if (underlying != nullptr) {
		TF_DeleteTensor(underlying);
	}
	underlying = nullptr;
}

template<TF_DataType DataTypeLabel>
typename Type<DataTypeLabel>::type& Tensor<DataTypeLabel>::at(int64_t const *indices, int64_t len)
{
	int64_t index = indices[len-1];
	int64_t multiplier = 1;

	for (int64_t i = len - 2; i >= 0; --i) {
		multiplier *= TF_Dim(underlying, i + 1);
		index += indices[i] * multiplier;
	}

	char* adr = (char*) TF_TensorData(underlying) + TF_DataTypeSize(DataTypeLabel) * index;
	
	return *(typename Type<DataTypeLabel>::type*)adr;
}

template<TF_DataType DataTypeLabel>
size_t Tensor<DataTypeLabel>::hash() const {
    int64_t len = 1;
    for (const int64_t &dim: dims) {
        len *= dim;
    }
    len *= TF_DataTypeSize(DataTypeLabel);

    char* data = (char*) TF_TensorData(underlying);
    size_t hash = std::hash<char>()(*data);
    for (int64_t i = 1; i < len; ++i) {
        ++data;
        hash = hash_combine(hash, *data);
    }

    return hash;
}

#endif //TFL_TENSOR_H
