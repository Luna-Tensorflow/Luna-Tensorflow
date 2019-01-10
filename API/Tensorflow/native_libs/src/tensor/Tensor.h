#ifndef TFL_TENSOR_H
#define TFL_TENSOR_H

#include <vector>
#include <tensorflow/c/c_api.h>
#include <cstring>
#include <cstddef>
#include <numeric>

#include "../helpers/utils.h"
#include "TypeLabel.h"

class TypeErasedTensor {
protected:
	 TF_Tensor *underlying;

	 TypeErasedTensor(TF_Tensor* tensor = nullptr);
public:
	 std::vector<int64_t> shape() const;

	 size_t flatSize() const;

	 TF_Tensor* get_underlying() const;

	 size_t hash() const;

	 virtual ~TypeErasedTensor();
};

template<TF_DataType DataTypeLabel>
class Tensor : public TypeErasedTensor {
private:
	using type = typename Type<DataTypeLabel>::tftype;
public:
	explicit Tensor(const type* vect, int64_t len);
	explicit Tensor(const type** array, int64_t width, int64_t height);

	explicit Tensor(const type *data, const int64_t *dims, int num_dims);

	explicit Tensor(TF_Tensor* underlying);

	Tensor(const Tensor& other);

	Tensor(Tensor&& other) noexcept;

	type& at(int64_t const* indices, int64_t len);
	type& at(const std::vector<int64_t> &indices);
};

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor<DataTypeLabel>::type *vect, int64_t len) : Tensor(vect, &len, 1) {
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor<DataTypeLabel>::type **array, int64_t width, int64_t height) {
	size_t data_size = TF_DataTypeSize(DataTypeLabel);
	auto dims = std::vector<int64_t>{width, height};
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
Tensor<DataTypeLabel>::Tensor(TF_Tensor* underlying) : TypeErasedTensor(underlying) {}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const type *data, const int64_t *dims, int num_dims) {
    size_t required_data_size = std::accumulate(dims, dims + num_dims, 1, [](int64_t a, int64_t b){return a * b;})
    		* TF_DataTypeSize(DataTypeLabel);
    underlying = TF_AllocateTensor(DataTypeLabel, dims, num_dims, required_data_size);
    memcpy(TF_TensorData(underlying), data, required_data_size);
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor<DataTypeLabel> &other)
{
	TF_Tensor *other_underlying = other.get_underlying();
	auto data_size = TF_TensorByteSize(other_underlying);
	auto dims = other.shape();
	underlying = TF_AllocateTensor(TF_TensorType(other_underlying), dims.data(), dims.size(), data_size);
	memcpy(TF_TensorData(underlying), TF_TensorData(other_underlying), data_size);
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(Tensor<DataTypeLabel> &&other) noexcept
{
	underlying = other.underlying;
	other.underlying = nullptr;
}

template<TF_DataType DataTypeLabel>
typename Tensor<DataTypeLabel>::type& Tensor<DataTypeLabel>::at(const std::vector<int64_t> &indices)
{
	return at(indices.data(), indices.size());
}

template<TF_DataType DataTypeLabel>
typename Tensor<DataTypeLabel>::type& Tensor<DataTypeLabel>::at(int64_t const *indices, int64_t len)
{
	int64_t index = indices[len-1];
	int64_t multiplier = 1;

	for (int64_t i = len - 2; i >= 0; --i) {
		multiplier *= TF_Dim(underlying, i + 1);
		index += indices[i] * multiplier;
	}

	char* adr = (char*) TF_TensorData(underlying) + TF_DataTypeSize(DataTypeLabel) * index;

	return *(typename Tensor<DataTypeLabel>::type*)adr;
}

#endif //TFL_TENSOR_H
