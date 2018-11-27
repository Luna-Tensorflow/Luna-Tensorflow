#include <numeric>
#include <cstring>

#include "Tensor.h"

Tensor::Tensor(const std::vector<int64_t> &dims, TF_DataType DataTypeLabel) {
    underlying = TF_AllocateTensor(DataTypeLabel, dims.data(), dims.size(), std::accumulate(dims.begin(), dims.end(), 1, [](int64_t a, int64_t b){return a * b;}) * TF_DataTypeSize(DataTypeLabel));
}

Tensor::Tensor(TF_Tensor* underlying) : underlying(underlying) {
}

Tensor::Tensor(const Tensor &other) {
    TF_Tensor *other_underlying = other.get_underlying();
    std::vector<int64_t> other_dims(TF_NumDims(other_underlying));
    int data_size = TF_TensorByteSize(other_underlying);
    for (int i = 0; i < other_dims.size(); ++i) {
        other_dims[i] = TF_Dim(other_underlying, i);
    }
    underlying = TF_AllocateTensor(TF_TensorType(other_underlying), other_dims.data(), other_dims.size(), data_size);
    memcpy(TF_TensorData(underlying), TF_TensorData(other_underlying), data_size);
}

Tensor::Tensor(Tensor&& other) {
    underlying = other.underlying;
    other.underlying = nullptr;
}

TF_Tensor* Tensor::get_underlying() const {
    return underlying;
}

Tensor::~Tensor() {
    if (underlying != nullptr) {
        TF_DeleteTensor(underlying);
    }
    underlying = nullptr;
}