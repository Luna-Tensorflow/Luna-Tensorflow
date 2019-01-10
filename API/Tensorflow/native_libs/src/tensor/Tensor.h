#ifndef TFL_TENSOR_H
#define TFL_TENSOR_H

#include <vector>
#include <tensorflow/c/c_api.h>
#include <cstring>
#include <cstddef>
#include <functional>

#include "../helpers/utils.h"
#include "TypeLabel.h"

class TypeErasedTensor {
protected:
	 TF_Tensor *underlying;

	 TypeErasedTensor(TF_Tensor* tensor = nullptr) : underlying(tensor) {}
public:
	 std::vector<int64_t> shape();

	 size_t flatSize();

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

	explicit Tensor(const std::vector<type> &vect);
	explicit Tensor(const std::vector<std::vector<type>> &array);

	explicit Tensor(TF_Tensor* underlying);

	Tensor(const Tensor& other);

	Tensor(Tensor&& other) noexcept;

	type& at(int64_t const* indices, int64_t len);
	type& at(const std::vector<int64_t> &indices);
};

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor::type *vect, int64_t len)
	: TypeErasedTensor() {
	size_t data_size = TF_DataTypeSize(DataTypeLabel);
	auto dims = std::vector<int64_t>{len};
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
	: TypeErasedTensor() {
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
Tensor<DataTypeLabel>::Tensor(const std::vector<Tensor::type> &vect) : Tensor(vect.data(), vect.size()) {}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const std::vector<std::vector<Tensor::type>> &array)
	: TypeErasedTensor() {
	size_t data_size = TF_DataTypeSize(DataTypeLabel);
	auto dims = std::vector<int64_t>{array.size(), array.front().size()};
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
Tensor<DataTypeLabel>::Tensor(TF_Tensor* underlying) : TypeErasedTensor(underlying) {}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(const Tensor &other)
{
	TF_Tensor *other_underlying = other.get_underlying();
	auto data_size = TF_TensorByteSize(other_underlying);
	auto dims = shape();
	underlying = TF_AllocateTensor(TF_TensorType(other_underlying), dims.data(), dims.size(), data_size);
	memcpy(TF_TensorData(underlying), TF_TensorData(other_underlying), data_size);
}

template<TF_DataType DataTypeLabel>
Tensor<DataTypeLabel>::Tensor(Tensor &&other) noexcept
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

template<>
class Tensor<TF_STRING> : public TypeErasedTensor {
private:
    //using type = typename Type<TF_STRING>::tftype;
    size_t flattenedLen;
    size_t getOffset(size_t idx) {
        return reinterpret_cast<uint64_t*>(TF_TensorData(underlying))[idx] + 8 * flattenedLen;
    }
    size_t getLength(size_t idx) {
        auto myOffset = reinterpret_cast<uint64_t*>(TF_TensorData(underlying))[idx] + 8 * flattenedLen;
        if (idx == flattenedLen - 1) {
            return TF_TensorByteSize(underlying) - myOffset;
        } else {
            auto nextOffset = reinterpret_cast<uint64_t*>(TF_TensorData(underlying))[idx + 1] + 8 * flattenedLen;
            return nextOffset - myOffset;
        }
    }
public:
    explicit Tensor(const char** vect, int64_t len) : TypeErasedTensor() {
        auto dims = std::vector<int64_t>{len};
        size_t nbytes = static_cast<size_t>(len * 8);
        for (int64_t i = 0; i < len; ++i) {
            nbytes += TF_StringEncodedSize(strlen(vect[i]));
        }

        underlying = TF_AllocateTensor(TF_STRING, dims.data(), 1, nbytes);

        uint64_t *offsets = static_cast<uint64_t *>(TF_TensorData(underlying));
        char* data = static_cast<char *>(TF_TensorData(underlying)) + len * 8;
        uint64_t offset = 0;
        for (int64_t i = 0; i < len; ++i) {
            offsets[i] = offset;
            auto slen = strlen(vect[i]);
            run_with_status<void>(std::bind(TF_StringEncode, vect[i], slen, data + offset, TF_StringEncodedSize(slen), std::placeholders::_1));
        }
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
};

#endif //TFL_TENSOR_H
