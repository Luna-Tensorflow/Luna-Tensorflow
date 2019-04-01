#include "tensorflow/c/c_api.h"
#include "tensors.h"
#include "../tensor/Tensor.h"
#include "../helpers/LifeTimeManager.h"
#include "../helpers/error.h"

#include <vector>
#include <memory>
#include <random>
#include <fstream>

template<typename T> T* vector_as_array(const std::vector<T>& vec) {
    T* r = static_cast<T*>(malloc(sizeof(T) * vec.size()));
    memcpy(r, vec.data(), sizeof(T) * vec.size());
    return r;
}

TFL_API Tensor *make_tensor(void const *array, TF_DataType type, const int64_t *dims, size_t num_dims, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(array, type, dims, num_dims);
        auto tensor_ptr = std::make_shared<Tensor>(array, dims, num_dims, type);
        auto t = LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
        FFILOGANDRETURN(t, array, type, dims, num_dims);
    };
}

TFL_API int get_tensor_num_dims(Tensor *tensor, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        auto r = LifetimeManager::instance().accessOwned(tensor)->shape().size();
        FFILOGANDRETURN(static_cast<int>(r), tensor);
    };
}

TFL_API int64_t get_tensor_dim(Tensor *tensor, int32_t dim_index, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        auto r = LifetimeManager::instance().accessOwned(tensor)->shape()[dim_index];
        FFILOGANDRETURN(r, tensor, dim_index);
    };
}

TFL_API int64_t* get_tensor_dims(Tensor *tensor, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        auto shape = LifetimeManager::instance().accessOwned(tensor)->shape();
        auto r = vector_as_array(shape);
        FFILOGANDRETURN(r, tensor);
    };
}

TFL_API int64_t get_tensor_flatlist_length(Tensor* tensor, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        int64_t l = LifetimeManager::instance().accessOwned(tensor)->flatSize();
        FFILOGANDRETURN(l, tensor);
    };
}

static inline void write_int64(std::ostream& binary_stream, int64_t x) {
    binary_stream.write(reinterpret_cast<char*>(&x), sizeof(x));
}

static inline int64_t read_int64(std::istream& binary_stream) {
    int64_t x;
    binary_stream.read(reinterpret_cast<char*>(&x), sizeof(x));
    return x;
}

static void write_tensor(std::ostream& binary_stream, std::shared_ptr<Tensor> tensor) {
    TF_Tensor* underlying = tensor->get_underlying();

    int64_t size = TF_TensorByteSize(underlying);
    write_int64(binary_stream, size);

    int64_t typetag = static_cast<int64_t >(TF_TensorType(underlying));
    write_int64(binary_stream, typetag);

    std::vector<int64_t> dims = tensor->shape();
    write_int64(binary_stream, dims.size());
    for (auto dim : dims) {
        write_int64(binary_stream, dim);
    }

    binary_stream.write(static_cast<const char*>(TF_TensorData(underlying)), size);
}

static std::shared_ptr<Tensor> read_tensor(std::istream& binary_stream) {
    int64_t size = read_int64(binary_stream);
    int64_t typetag_value = read_int64(binary_stream);
    TF_DataType typetag = static_cast<TF_DataType>(typetag_value);
    int64_t dims_length = read_int64(binary_stream);
    std::vector<int64_t> dims(static_cast<unsigned long>(dims_length));
    for (auto& dim : dims) {
        dim = read_int64(binary_stream);
    }

    TF_Tensor* underlying = TF_AllocateTensor(typetag, dims.data(), static_cast<int>(dims.size()), static_cast<size_t>(size));
    try {
        binary_stream.read(reinterpret_cast<char*>(TF_TensorData(underlying)), size);
        return std::make_shared<Tensor>(underlying);
    } catch(...) {
        TF_DeleteTensor(underlying); // if allocation failed, we free the tensor
        throw;
    }
}

TFL_API void save_tensors_to_file(const char* filename, Tensor** tensors, int64_t count, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(filename, tensors, count);
        std::ofstream file(filename, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Error opening file " + std::string(filename) + " for writing");
        }

        write_int64(file, count);
        for (int64_t i = 0; i < count; ++i) {
            write_tensor(file, LifetimeManager::instance().accessOwned(tensors[i]));
        }
    };
}

TFL_API Tensor** load_tensors_from_file(const char* filename, int64_t count, const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG(filename, count);
        std::ifstream file(filename, std::ios::in | std::ios::binary);
        if (!file.good()) {
            throw std::runtime_error("Error opening file " + std::string(filename) + " for reading");
        }

        int64_t true_count = read_int64(file);
        if (true_count != count) {
            throw std::runtime_error("File contains " + std::to_string(true_count) + " tensors, but runtime wanted to load " + std::to_string(count));
        }

        std::vector<std::shared_ptr<Tensor>> result(static_cast<unsigned long>(count));
        for (int64_t i = 0; i < count; ++i) {
            result[i] = read_tensor(file);
        }

        return LifetimeManager::instance().addOwnershipOfArray(result);
    };
}

#define GET_TENSOR_VALUE_AT(typelabel) \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_##typelabel(Tensor *tensor, int64_t *idxs, size_t len, const char **outError) { \
    return TRANSLATE_EXCEPTION(outError) { \
        auto r = LifetimeManager::instance().accessOwned(tensor)->at<typelabel>(idxs, len); \
        FFILOGANDRETURN(r, tensor, idxs, len); \
    }; \
}

#define GET_TENSOR_VALUE_AT_INDEX(typelabel) \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_index_##typelabel(Tensor *tensor, int64_t index, const char **outError) { \
    return TRANSLATE_EXCEPTION(outError) { \
        auto r = LifetimeManager::instance().accessOwned(tensor)->at<typelabel>(index); \
        FFILOGANDRETURN(r, tensor, index); \
    }; \
}

#define TENSOR_TO_FLATLIST(typelabel) \
TFL_API Type<typelabel>::lunatype* tensor_to_flatlist_##typelabel(Tensor* tensor, const char **outError) { \
    return TRANSLATE_EXCEPTION(outError) { \
        auto t = LifetimeManager::instance().accessOwned(tensor); \
        auto len = t->flatSize(); \
        Type<typelabel>::lunatype* r = static_cast<Type<typelabel>::lunatype*> (malloc(sizeof(Type<typelabel>::lunatype) * len)); \
        for (size_t i = 0; i < len; ++i) { \
            r[i] = t->at<typelabel>(i); \
        } \
        FFILOGANDRETURN(r, tensor); \
    }; \
}

#define MAKE_RANDOM_TENSOR(typelabel, type) \
TFL_API Tensor *make_random_tensor_##typelabel(const int64_t *dims, size_t num_dims, \
    Type<typelabel>::lunatype const min, Type<typelabel>::lunatype const max, const char **outError){ \
    return TRANSLATE_EXCEPTION(outError) { \
        FFILOG(dims, num_dims, min, max); \
        int64_t elems = std::accumulate(dims, dims+num_dims, 1, std::multiplies<int64_t>()); \
        auto* data = (Type<typelabel>::tftype*) malloc(elems * TF_DataTypeSize(typelabel)); \
        \
        std::mt19937 engine(std::random_device{}()); \
        std::uniform_##type##_distribution distribution(min, max); \
        \
        std::generate(data, data+elems, [&]() { \
                return distribution(engine); \
        }); \
        \
        auto tensor = std::make_shared<Tensor>(data, dims, num_dims, typelabel); \
        free(data); \
        \
        return LifetimeManager::instance().addOwnership(std::move(tensor));\
    }; \
}

#define MAKE_CONST_TENSOR(typelabel, type) \
TFL_API Tensor *make_const_tensor_##typelabel(const int64_t *dims, size_t num_dims, \
	Type<typelabel>::lunatype const value, const char **outError){ \
	return TRANSLATE_EXCEPTION(outError) { \
	    FFILOG(dims, num_dims, value); \
        int64_t elems = std::accumulate(dims, dims+num_dims, 1, std::multiplies<int64_t>()); \
        auto* data = (Type<typelabel>::tftype*) malloc(elems * TF_DataTypeSize(typelabel)); \
        \
        std::generate(data, data+elems, [&]() { \
                return value; \
        }); \
        \
        auto tensor = std::make_shared<Tensor>(data, dims, num_dims, typelabel); \
        free(data); \
        \
        return LifetimeManager::instance().addOwnership(std::move(tensor));\
    }; \
}



#define DEFINE_TENSOR(typelabel) \
GET_TENSOR_VALUE_AT(typelabel); \
GET_TENSOR_VALUE_AT_INDEX(typelabel); \
TENSOR_TO_FLATLIST(typelabel);

#define DEFINE_TENSOR_NUMERIC(typelabel, randomtype) \
DEFINE_TENSOR(typelabel); \
MAKE_RANDOM_TENSOR(typelabel, randomtype);\
MAKE_CONST_TENSOR(typelabel, randomtype);\


DEFINE_TENSOR_NUMERIC(TF_FLOAT, real);
DEFINE_TENSOR_NUMERIC(TF_DOUBLE, real);
DEFINE_TENSOR_NUMERIC(TF_INT8, int);
DEFINE_TENSOR_NUMERIC(TF_INT16, int);
DEFINE_TENSOR_NUMERIC(TF_INT32, int);
DEFINE_TENSOR_NUMERIC(TF_INT64, int);
DEFINE_TENSOR_NUMERIC(TF_UINT8, int);
DEFINE_TENSOR_NUMERIC(TF_UINT16, int);
DEFINE_TENSOR_NUMERIC(TF_UINT32, int);
DEFINE_TENSOR_NUMERIC(TF_UINT64, int);
DEFINE_TENSOR(TF_BOOL);
DEFINE_TENSOR(TF_STRING);
//DEFINE_TENSOR(TF_HALF);
