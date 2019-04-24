#include "Tensor.h"

Tensor::Tensor(const void* vect, int64_t len, TF_DataType type) : Tensor(vect, &len, 1, type) {
}

Tensor::Tensor(const void *data, const int64_t *dims, int num_dims, TF_DataType type) : type(type) {
    flattenedLen = static_cast<size_t>(std::accumulate(dims, dims + num_dims, 1, std::multiplies<>()));
    if (type != TF_STRING) {
        underlying = TF_AllocateTensor(type, dims, num_dims, flattenedLen * TF_DataTypeSize(type));
        memcpy(TF_TensorData(underlying), data, flattenedLen * TF_DataTypeSize(type));
    } else {
        size_t n_bytes = flattenedLen * sizeof(uint64_t);
        for (size_t i = 0; i < flattenedLen; ++i) {
            n_bytes += TF_StringEncodedSize(strlen(static_cast<const char * const *>(data)[i]));
        }

        underlying = TF_AllocateTensor(TF_STRING, dims, num_dims, n_bytes);

        uint64_t *offsets = static_cast<uint64_t*>(TF_TensorData(underlying));
        char* elem_data = static_cast<char*>(TF_TensorData(underlying)) + flattenedLen * 8;
        uint64_t offset = 0;
        for (size_t i = 0; i < flattenedLen; ++i) {
            offsets[i] = offset;
            auto slen = strlen(static_cast<const char * const *>(data)[i]);
            auto encoded_len = TF_StringEncodedSize(slen);
            run_with_status<void>(std::bind(TF_StringEncode, static_cast<const char * const *>(data)[i], slen, elem_data + offset, encoded_len, std::placeholders::_1));
            offset += encoded_len;
        }
    }
}

Tensor::Tensor(const void *data, const std::vector<int64_t> &dims, TF_DataType type) : Tensor(data, dims.data(), dims.size(), type) {
}

Tensor::Tensor(TF_Tensor* underlying) : underlying(underlying), type(TF_TensorType(underlying)), flattenedLen(flatSize()) {
}

Tensor::Tensor(const Tensor& other) { // Not delegating the constructor here because of TF_STRING handling
    type = other.type;
    TF_Tensor *other_underlying = other.get_underlying();
    auto data_size = TF_TensorByteSize(other_underlying);
    auto dims = other.shape();
    underlying = TF_AllocateTensor(TF_TensorType(other_underlying), dims.data(), dims.size(), data_size);
    memcpy(TF_TensorData(underlying), TF_TensorData(other_underlying), data_size);
    flattenedLen = flatSize();
}

Tensor::Tensor(Tensor&& other) noexcept {
    type = other.type;
    underlying = other.underlying;
    other.underlying = nullptr;
    flattenedLen = flatSize();
    other.flattenedLen = 0;
}

std::vector<int64_t> Tensor::shape() const {
    int ndims = TF_NumDims(underlying);
    std::vector<int64_t> dims(ndims);
    for (int i = 0; i < ndims; ++i) {
        dims[i] = TF_Dim(underlying, i);
    }
    return dims;
}

size_t Tensor::flatSize() const {
    auto dims = shape();
    return static_cast<size_t>(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>()));
}

TF_Tensor* Tensor::get_underlying() const {
    return underlying;
}

size_t Tensor::hashcode() const {
    size_t bytes = TF_TensorByteSize(underlying);

    char* data = (char*) TF_TensorData(underlying);
    size_t hash = std::hash<char>()(*data);
    for (size_t i = 1; i < bytes; ++i) {
        ++data;
        hash = hash_combine(hash, *data);
    }

    return hash;
}

Tensor::~Tensor() {
    LOG("deleting tensor ", vec_to_string(shape()));

    if (underlying != nullptr) {
        TF_DeleteTensor(underlying);
    }
    underlying = nullptr;
}

template<>
Type<TF_STRING>::tfattype Tensor::at<TF_STRING>(int64_t index) {
    auto offset = getOffset(static_cast<size_t>(index));
    auto encoded_len = getLength(static_cast<size_t>(index));

    const char* str;
    size_t decoded_len;
    run_with_status<void>(std::bind(TF_StringDecode, reinterpret_cast<const char*>(TF_TensorData(underlying)) + offset, encoded_len, &str, &decoded_len, std::placeholders::_1));

    char* cpy = static_cast<char *>(malloc(decoded_len + 1));
    memcpy(cpy, str, decoded_len);
    cpy[decoded_len] = 0;
    return cpy;
}

TF_DataType Tensor::getType() const {
    return type;
}

size_t Tensor::getOffset(size_t idx) const {
    if (type != TF_STRING) {
        return idx;
    } else {
        return reinterpret_cast<uint64_t *>(TF_TensorData(underlying))[idx] + 8 * flattenedLen;
    }
}
size_t Tensor::getLength(size_t idx) const {
    if (type != TF_STRING) {
        return TF_DataTypeSize(type);
    } else {
        auto myOffset = getOffset(idx);
        if (idx == flattenedLen - 1) {
            return TF_TensorByteSize(underlying) - myOffset;
        } else {
            auto nextOffset = getOffset(idx + 1);
            return nextOffset - myOffset;
        }
    }
}