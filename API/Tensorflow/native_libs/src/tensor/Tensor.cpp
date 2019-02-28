#include "Tensor.h"

TypeErasedTensor::TypeErasedTensor(TF_Tensor* tensor) : underlying(tensor) {}

std::vector<int64_t> TypeErasedTensor::shape() const {
    int ndims = TF_NumDims(underlying);
    std::vector<int64_t> dims(ndims);
    for (int i = 0; i < ndims; ++i) {
        dims[i] = TF_Dim(underlying, i);
    }
    return dims;
}

size_t TypeErasedTensor::flatSize() const {
    size_t r = 1;
    for (auto dim : shape()) {
        r *= dim;
    }
    return r;
}

TF_Tensor* TypeErasedTensor::get_underlying() const {
    return underlying;
}

size_t TypeErasedTensor::hash() const {
    size_t bytes = TF_TensorByteSize(underlying);

    char* data = (char*) TF_TensorData(underlying);
    size_t hash = std::hash<char>()(*data);
    for (size_t i = 1; i < bytes; ++i) {
        ++data;
        hash = hash_combine(hash, *data);
    }

    return hash;
}

TypeErasedTensor::~TypeErasedTensor() {
    if (underlying != nullptr) {
        TF_DeleteTensor(underlying);
    }
    underlying = nullptr;
}