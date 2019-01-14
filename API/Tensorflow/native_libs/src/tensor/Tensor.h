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

	explicit Tensor(const type *data, const int64_t *dims, int num_dims);

	explicit Tensor(const type *data, const std::vector<int64_t> &dims);

	explicit Tensor(TF_Tensor* underlying);

	Tensor(const Tensor& other);

	Tensor(Tensor&& other) noexcept;

	type& at(int64_t const* indices, int64_t len);
	type& at(const std::vector<int64_t> &indices);
	type& at(int64_t index);
};

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor<DataTypeLabel>::type *vect, int64_t len) : Tensor(vect, &len, 1) {
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
Tensor<DataTypeLabel>::Tensor(const type *data, const std::vector<int64_t> &dims)
        : Tensor(data, dims.data(), dims.size()) {
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor<DataTypeLabel> &other)
        : Tensor(TF_TensorData(other.get_underlying()), other.shape()) {
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
	std::vector<int64_t> dims = shape();

	for (int64_t i = len - 2; i >= 0; --i) {
		multiplier *= dims[i + 1];
		index += indices[i] * multiplier;
	}

	return at(index);
}

template<TF_DataType DataTypeLabel>
typename Tensor<DataTypeLabel>::type& Tensor<DataTypeLabel>::at(int64_t index)
{
	char* adr = (char*) TF_TensorData(underlying) + TF_DataTypeSize(DataTypeLabel) * index;
	return *(typename Tensor<DataTypeLabel>::type*)adr;
}

template<>
class Tensor<TF_STRING> : public TypeErasedTensor {
private:
    //using type = typename Type<TF_STRING>::tftype;
    size_t flattenedLen;
    size_t getOffset(size_t idx) {
        return reinterpret_cast<uint64_t*>(TF_TensorData(underlying))[idx] + 8 * flattenedLen;
    }
    size_t getLength(size_t idx) {
        auto myOffset = getOffset(idx);
        if (idx == flattenedLen - 1) {
            return TF_TensorByteSize(underlying) - myOffset;
        } else {
            auto nextOffset = getOffset(idx + 1);
            return nextOffset - myOffset;
        }
    }
public:
    explicit Tensor(const char **vect, int64_t len) : Tensor(vect, &len, 1) {
    }

    explicit Tensor(const char **data, const int64_t *dims, int num_dims) : TypeErasedTensor() {
        flattenedLen = static_cast<size_t>(std::accumulate(dims, dims + num_dims, 1, [](int64_t a, int64_t b){return a * b;}));
        size_t n_bytes = flattenedLen * sizeof(uint64_t);
        for (size_t i = 0; i < flattenedLen; ++i) {
            n_bytes += TF_StringEncodedSize(strlen(data[i]));
        }

        underlying = TF_AllocateTensor(TF_STRING, dims, num_dims, n_bytes);

        uint64_t *offsets = static_cast<uint64_t*>(TF_TensorData(underlying));
        char* elem_data = static_cast<char*>(TF_TensorData(underlying)) + flattenedLen * 8;
        uint64_t offset = 0;
        for (size_t i = 0; i < flattenedLen; ++i) {
            offsets[i] = offset;
            auto slen = strlen(data[i]);
            auto encoded_len = TF_StringEncodedSize(slen);
            run_with_status<void>(std::bind(TF_StringEncode, data[i], slen, elem_data + offset, encoded_len, std::placeholders::_1));
            offset += encoded_len;
        }
    }

    explicit Tensor(const char **data, const std::vector<int64_t> &dims) : Tensor(data, dims.data(), dims.size()) {
    }

    explicit Tensor(TF_Tensor* underlying) : TypeErasedTensor(underlying) {
        flattenedLen = flatSize();
    }

    explicit Tensor(const Tensor& other) : TypeErasedTensor() {
        TF_Tensor *other_underlying = other.get_underlying();
        auto data_size = TF_TensorByteSize(other_underlying);
        auto dims = shape();
        underlying = TF_AllocateTensor(TF_TensorType(other_underlying), dims.data(), dims.size(), data_size);
        memcpy(TF_TensorData(underlying), TF_TensorData(other_underlying), data_size);
        flattenedLen = flatSize();
    }

    Tensor(Tensor&& other) noexcept {
        underlying = other.underlying;
        other.underlying = nullptr;
        flattenedLen = flatSize();
        other.flattenedLen = 0;
    }

    /*
     * at allocates a copy of the string and the caller assumes ownership over it, so they should free it
     */
    char* at(int64_t const* indices, int64_t len) {
        int64_t index = indices[len-1];
        int64_t multiplier = 1;

        for (int64_t i = len - 2; i >= 0; --i) {
            multiplier *= TF_Dim(underlying, i + 1);
            index += indices[i] * multiplier;
        }

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

    char* at(const std::vector<int64_t> &indices) {
        return at(indices.data(), indices.size());
    }

    char* at(int64_t index) {
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
};

#endif //TFL_TENSOR_H
