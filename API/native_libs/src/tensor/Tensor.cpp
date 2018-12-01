#include <numeric>
#include <cstring>

#include "Tensor.h"


template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const type* vect, size_t len)
{
	size_t data_size = TF_DataTypeSize(DataTypeLabel);
	dims = std::vector<size_t>{len};
	underlying = TF_AllocateTensor(DataTypeLabel, dims.data(), 1, data_size * len);

	auto* adr = TF_TensorData(underlying);
	for(auto i=0; i<len; ++i)
	{
		*adr = vect[i];
		adr += data_size;
	}
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const type** array, size_t width, size_t height)
{
	size_t data_size = TF_DataTypeSize(DataTypeLabel);
	dims = std::vector<size_t>{width, height};
	underlying = TF_AllocateTensor(DataTypeLabel, dims.data(), 2, width * height * data_size);

	auto* adr = TF_TensorData(underlying);
	for(auto i=0; i<width; ++i)
	{
		for(auto j=0; j<height; ++j)
		{
			*adr = array[i][j];
			adr += data_size;
		}
	}
}

template <TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const std::vector<std::vector<Tensor::type>> &array)
{
	size_t data_size = TF_DataTypeSize(DataTypeLabel);
	dims = std::vector<size_t>{array.size(), array.front().size()};
	underlying = TF_AllocateTensor(DataTypeLabel, dims.data(), 2, array.size() * array.front().size() * data_size);

	auto* adr = TF_TensorData(underlying);
	for(auto i=0; i<array.size(); ++i)
	{
		for(auto j=0; j<array[i].size(); ++j)
		{
			*adr = array[i][j];
			adr += data_size;
		}
	}
}

template <TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(TF_Tensor* underlying) : underlying(underlying) {}

template <TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor<DataTypeLabel> &other) {
	TF_Tensor *other_underlying = other.get_underlying();
	std::vector<int64_t> other_dims(TF_NumDims(other_underlying));
	auto data_size = TF_TensorByteSize(other_underlying);
	for (int i = 0; i < other_dims.size(); ++i) {
		other_dims[i] = TF_Dim(other_underlying, i);
	}
	underlying = TF_AllocateTensor(TF_TensorType(other_underlying), other_dims.data(), other_dims.size(), data_size);
	memcpy(TF_TensorData(underlying), TF_TensorData(other_underlying), data_size);
}

template <TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(Tensor<DataTypeLabel>&& other) noexcept {
	underlying = other.underlying;
	other.underlying = nullptr;
}

template <TF_DataType DataTypeLabel>
TF_Tensor* Tensor<DataTypeLabel>::get_underlying() const {
	return underlying;
}


template <TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::~Tensor() {
	if (underlying != nullptr) {
		TF_DeleteTensor(underlying);
	}
	underlying = nullptr;
}

template<TF_DataType DataTypeLabel>
typename Type<DataTypeLabel>::type &Tensor<DataTypeLabel>::at(const std::vector<int64_t> &indices)
{
	int64_t index = indices.back();
	int64_t multiplier = 1;

	for (size_t i = indices.size() - 2; i >= 0; --i) {
		multiplier *= TF_Dim(underlying, i + 1);
		index += indices[i] * multiplier;
	}

	char* adr = (char*) TF_TensorData(underlying) + TF_DataTypeSize(DataTypeLabel) * index;

	return *(typename Type<DataTypeLabel>::type*)adr;
}

template<TF_DataType DataTypeLabel>
std::vector<size_t> Tensor<DataTypeLabel>::shape()
{
	return dims;
}


