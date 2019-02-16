#ifndef TFL_TENSOR_H
#define TFL_TENSOR_H

#include <vector>
#include <tensorflow/c/c_api.h>
#include <cstring>
#include <cstddef>
#include <functional>
#include <numeric>

#include "../helpers/utils.h"
#include "TypeLabel.h"

class Tensor {
private:
    TF_Tensor *underlying;
    TF_DataType type;
    size_t flattenedLen;

public:
    explicit Tensor(const void* vect, int64_t len, TF_DataType type);

    explicit Tensor(const void *data, const int64_t *dims, int num_dims, TF_DataType type);

    explicit Tensor(const void *data, const std::vector<int64_t> &dims, TF_DataType type);

    explicit Tensor(TF_Tensor* underlying);

    Tensor(const Tensor& other);

    Tensor(Tensor&& other) noexcept;

    ~Tensor();

    template<TF_DataType DataTypeLabel>
    typename Type<DataTypeLabel>::tfattype at(int64_t const* indices, int64_t len);

    template<TF_DataType DataTypeLabel>
    typename Type<DataTypeLabel>::tfattype at(const std::vector<int64_t> &indices);

    template<TF_DataType DataTypeLabel>
    typename Type<DataTypeLabel>::tfattype at(int64_t index);

    std::vector<int64_t> shape() const;

    size_t flatSize() const;

    TF_Tensor* get_underlying() const;

    size_t hash() const;

    size_t getOffset(size_t idx) const;
    size_t getLength(size_t idx) const;
};

template<TF_DataType DataTypeLabel>
typename Type<DataTypeLabel>::tfattype Tensor::at(int64_t const* indices, int64_t len) {
    int64_t index = indices[len-1];
    int64_t multiplier = 1;
    std::vector<int64_t> dims = shape();

    for (int64_t i = len - 2; i >= 0; --i) {
        multiplier *= dims[i + 1];
        index += indices[i] * multiplier;
    }

    return at<DataTypeLabel>(index);
}

template<TF_DataType DataTypeLabel>
typename Type<DataTypeLabel>::tfattype Tensor::at(const std::vector<int64_t> &indices) {
    return at<DataTypeLabel>(indices.data(), indices.size());
}

template<TF_DataType DataTypeLabel>
typename Type<DataTypeLabel>::tfattype Tensor::at(int64_t index) {
    char* adr = (char*) TF_TensorData(underlying) + TF_DataTypeSize(DataTypeLabel) * index;
    return *(typename Type<DataTypeLabel>::tftype*)adr;
}

template<>
typename Type<TF_STRING>::tfattype Tensor::at<TF_STRING>(int64_t index);

#endif //TFL_TENSOR_H
