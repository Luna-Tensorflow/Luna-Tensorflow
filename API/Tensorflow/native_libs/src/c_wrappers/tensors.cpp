//
// Created by mateusz on 01.12.18.
//

#include "tensorflow/c/c_api.h"
#include "tensors.h"
#include "../tensor/Tensor.h"
#include "../helpers/LifeTimeManager.h"

#include <vector>
#include <memory>
#include <random>

TFL_API Tensor *make_tensor(void const *array, TF_DataType type, const int64_t *dims, size_t num_dims) {
    FFILOG(array, type, dims, num_dims);
	auto tensor_ptr = std::make_shared<Tensor>(array, dims, num_dims, type);
	auto t = LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
	FFILOGANDRETURN(t, array, type, dims, num_dims);
}

TFL_API int get_tensor_num_dims(Tensor *tensor) {
    auto r = LifetimeManager::instance().accessOwned(tensor)->shape().size();
    FFILOGANDRETURN(r, tensor);
}

TFL_API int64_t get_tensor_dim(Tensor *tensor, int32_t dim_index) {
    auto r = LifetimeManager::instance().accessOwned(tensor)->shape()[dim_index];
    FFILOGANDRETURN(r, tensor, dim_index);
}

#define GET_TENSOR_VALUE_AT(typelabel) \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_##typelabel(Tensor *tensor, int64_t *idxs, size_t len) { \
    auto r = LifetimeManager::instance().accessOwned(tensor)->at<typelabel>(idxs, len); \
    FFILOGANDRETURN(r, tensor, idxs, len); \
}

#define GET_TENSOR_VALUE_AT_INDEX(typelabel) \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_index_##typelabel(Tensor *tensor, int64_t index) { \
    auto r = LifetimeManager::instance().accessOwned(tensor)->at<typelabel>(index); \
    FFILOGANDRETURN(r, tensor, index); \
}

#define MAKE_RANDOM_TENSOR(typelabel, type) \
TFL_API Tensor *make_random_tensor_##typelabel(const int64_t *dims, size_t num_dims, \
	Type<typelabel>::lunatype const min, Type<typelabel>::lunatype const max){ \
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
}\



#define DECLARE_TENSOR(typelabel) \
GET_TENSOR_VALUE_AT(typelabel); \
GET_TENSOR_VALUE_AT_INDEX(typelabel); \

#define DECLARE_TENSOR_NUMERIC(typelabel, type) \
DECLARE_TENSOR(typelabel); \
MAKE_RANDOM_TENSOR(typelabel, type);\


DECLARE_TENSOR_NUMERIC(TF_FLOAT, real);
DECLARE_TENSOR_NUMERIC(TF_DOUBLE, real);
DECLARE_TENSOR_NUMERIC(TF_INT8, int);
DECLARE_TENSOR_NUMERIC(TF_INT16, int);
DECLARE_TENSOR_NUMERIC(TF_INT32, int);
DECLARE_TENSOR_NUMERIC(TF_INT64, int);
DECLARE_TENSOR_NUMERIC(TF_UINT8, int);
DECLARE_TENSOR_NUMERIC(TF_UINT16, int);
DECLARE_TENSOR_NUMERIC(TF_UINT32, int);
DECLARE_TENSOR_NUMERIC(TF_UINT64, int);
DECLARE_TENSOR(TF_BOOL);
DECLARE_TENSOR(TF_STRING);
//DECLARE_TENSOR(TF_HALF);