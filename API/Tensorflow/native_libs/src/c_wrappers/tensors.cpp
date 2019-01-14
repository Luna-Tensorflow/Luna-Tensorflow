//
// Created by mateusz on 01.12.18.
//

#include "tensors.h"
#include "../tensor/Tensor.h"
#include "../helpers/LifeTimeManager.h"

#include <vector>
#include <memory>
#include <numeric> //TODO

Tensor<TF_FLOAT> *make_float_tensor(float const* array, int64_t len)
{
	LOG(array, len);
	auto tensor_ptr = std::make_shared<Tensor<TF_FLOAT>>(array, len);

	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}

Tensor<TF_INT32> *make_int_tensor(const int32_t* array, int64_t len)
{
    LOG(array, len);
	auto tensor_ptr = std::make_shared<Tensor<TF_INT32>>(array, len);

	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}

float get_tensor1d_float_value_at(Tensor<TF_FLOAT> *tensor, int64_t idx)
{
	auto r = get_tensor_float_value_at(tensor, &idx, 1);
    LOGANDRETURN(r, tensor, idx);
}

float get_tensor_float_value_at(Tensor<TF_FLOAT> *tensor, int64_t *idxs, size_t len)
{
	auto r = LifetimeManager::instance().accessOwned(tensor)->at(idxs, len);
	LOGANDRETURN(r, tensor, idxs, len);
}

int32_t get_tensor1d_int_value_at(Tensor<TF_INT32> *tensor, int64_t idx)
{
	auto r = get_tensor_int_value_at(tensor, &idx, 1);
    LOGANDRETURN(r, tensor, idx);
}

int32_t get_tensor_int_value_at(Tensor<TF_INT32> *tensor, int64_t *idxs, size_t len)
{
	auto r = LifetimeManager::instance().accessOwned(tensor)->at(idxs, len);
    LOGANDRETURN(r, tensor, idxs, len);
}

int64_t tensor_float_length(Tensor<TF_FLOAT> * tensor) {
	auto r = LifetimeManager::instance().accessOwned(tensor)->shape()[0];
   LOGANDRETURN(r, tensor);
}

#define MAKE_TENSOR(typelabel) \
TFL_API Tensor<typelabel> *make_tensor_##typelabel(Type<typelabel>::lunatype const *array, const int64_t *dims, size_t num_dims) { \
    LOG(array, dims, num_dims); \
    static_assert(sizeof(Type<typelabel>::tftype) == sizeof(Type<typelabel>::lunatype), "tftype and lunatype need to be of same size"); \
    auto casted = reinterpret_cast<const Type<(typelabel)>::tftype*>(array); \
	auto tensor_ptr = std::make_shared<Tensor<(typelabel)>>(casted, dims, num_dims); \
	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr)); \
}

#define GET_TENSOR_VALUE_AT(typelabel) \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_##typelabel(Tensor<typelabel> *tensor, int64_t *idxs, size_t len) { \
    auto r = LifetimeManager::instance().accessOwned(tensor)->at(idxs, len); \
    LOGANDRETURN(r, tensor, idxs, len); \
}

#define GET_TENSOR_VALUE_AT_INDEX(typelabel) \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_index_##typelabel(Tensor<typelabel> *tensor, int64_t index) { \
    auto r = LifetimeManager::instance().accessOwned(tensor)->at(index); \
    LOGANDRETURN(r, tensor, index); \
}

#define GET_TENSOR_NUM_DIMS(typelabel) \
TFL_API int get_tensor_num_dims_##typelabel(Tensor<typelabel> *tensor) { \
    auto r = LifetimeManager::instance().accessOwned(tensor)->shape().size(); \
    LOGANDRETURN(r, tensor); \
}

#define GET_TENSOR_DIM(typelabel) \
TFL_API int64_t get_tensor_dim_##typelabel(Tensor<typelabel> *tensor, int32_t dim_index) { \
    auto r = LifetimeManager::instance().accessOwned(tensor)->shape()[dim_index]; \
    LOGANDRETURN(r, tensor, dim_index); \
}

#define DECLARE_TENSOR(typelabel) \
MAKE_TENSOR(typelabel); \
GET_TENSOR_VALUE_AT(typelabel); \
GET_TENSOR_VALUE_AT_INDEX(typelabel); \
GET_TENSOR_NUM_DIMS(typelabel); \
GET_TENSOR_DIM(typelabel);

DECLARE_TENSOR(TF_FLOAT);
DECLARE_TENSOR(TF_DOUBLE);
DECLARE_TENSOR(TF_INT8);
DECLARE_TENSOR(TF_INT16);
DECLARE_TENSOR(TF_INT32);
DECLARE_TENSOR(TF_INT64);
DECLARE_TENSOR(TF_UINT8);
DECLARE_TENSOR(TF_UINT16);
DECLARE_TENSOR(TF_UINT32);
DECLARE_TENSOR(TF_UINT64);
DECLARE_TENSOR(TF_BOOL);
//DECLARE_TENSOR(TF_HALF);

GET_TENSOR_VALUE_AT(TF_STRING);
GET_TENSOR_VALUE_AT_INDEX(TF_STRING);
GET_TENSOR_NUM_DIMS(TF_STRING);
GET_TENSOR_DIM(TF_STRING);
TFL_API Tensor<TF_STRING> *make_tensor_TF_STRING(Type<TF_STRING>::lunatype const *array, const int64_t *dims, size_t num_dims) {
   LOG(array, len);
	auto tensor_ptr = std::make_shared<Tensor<(TF_STRING)>>(const_cast<const char**>(array), dims, num_dims);
	// It's much easier to just free the C-strings here than to add rule exceptions in Luna.
	auto len = std::accumulate(dims, dims + num_dims, 1, [](int64_t a, int64_t b){return a * b;});
	for (int64_t i = 0; i < len; ++i) {
		free(array[i]);
	}
	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}