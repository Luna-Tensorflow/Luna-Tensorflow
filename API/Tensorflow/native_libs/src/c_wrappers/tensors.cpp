//
// Created by mateusz on 01.12.18.
//

#include "tensors.h"
#include "../tensor/Tensor.h"
#include "../helpers/LifeTimeManager.h"

#include <vector>
#include <memory>

Tensor<TF_FLOAT> *make_float_tensor(float const* array, int64_t len)
{
	LOG(array, len);
	auto tensor_ptr = std::make_shared<Tensor<TF_FLOAT>>(array, len);

	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}

Tensor<TF_FLOAT> *make_float_tensor_arr(float const** array, int64_t width, int64_t height)
{
   LOG(array, width, height);
	auto tensor_ptr = std::make_shared<Tensor<TF_FLOAT>>(array, width, height);

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
TFL_API Tensor<typelabel> *make_tensor_##typelabel(Type<typelabel>::lunatype const *array, int64_t len) { \
	LOG(array, len); \
   static_assert(sizeof(Type<typelabel>::tftype) == sizeof(Type<typelabel>::lunatype), "tftype and lunatype need to be of same size"); \
   auto casted = reinterpret_cast<const Type<(typelabel)>::tftype*>(array); \
	auto tensor_ptr = std::make_shared<Tensor<(typelabel)>>(casted, len); \
	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr)); \
}

#define MAKE_TENSOR_ARR(typelabel) \
TFL_API Tensor<typelabel> *make_tensor_arr_##typelabel(Type<typelabel>::lunatype const **array, int64_t width, int64_t height) { \
   LOG(array, width, height); \
	static_assert(sizeof(Type<typelabel>::tftype) == sizeof(Type<typelabel>::lunatype), "tftype and lunatype need to be of same size"); \
   auto casted = reinterpret_cast<const Type<(typelabel)>::tftype**>(array); \
	auto tensor_ptr = std::make_shared<Tensor<(typelabel)>>(casted, width, height); \
	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr)); \
}

#define GET_TENSOR_VALUE_AT(typelabel) \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_##typelabel(Tensor<typelabel> *tensor, int64_t *idxs, size_t len) { \
    auto r = LifetimeManager::instance().accessOwned(tensor)->at(idxs, len); \
    LOGANDRETURN(r, tensor, idxs, len); \
}

#define GET_TENSOR_LENGTH(typelabel) \
TFL_API int64_t get_tensor_length_##typelabel(Tensor<typelabel> *tensor) { \
    auto r = LifetimeManager::instance().accessOwned(tensor)->shape()[0]; \
    LOGANDRETURN(r, tensor); \
}

#define DECLARE_TENSOR(typelabel) \
MAKE_TENSOR(typelabel); \
MAKE_TENSOR_ARR(typelabel); \
GET_TENSOR_VALUE_AT(typelabel); \
GET_TENSOR_LENGTH(typelabel);

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
//DECLARE_TENSOR(TF_STRING);
//DECLARE_TENSOR(TF_HALF);