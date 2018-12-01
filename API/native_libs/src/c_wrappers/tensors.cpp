//
// Created by mateusz on 01.12.18.
//

#include "tensors.h"
#include <API/native_libs/src/tensor/Tensor.h>
#include <API/native_libs/src/LifeTimeManager.h>

#include <vector>
#include <memory>

Tensor<TF_FLOAT> *make_float_tensor(float const* array, int64_t len)
{
	Tensor<TF_FLOAT> tensor(array, len);
	auto tensor_ptr = std::make_shared<Tensor<TF_FLOAT>>(tensor);

	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}

Tensor<TF_FLOAT> *make_float_tensor_arr(float const** array, int64_t width, int64_t height)
{
	Tensor<TF_FLOAT> tensor(array, width, height);
	auto tensor_ptr = std::make_shared<Tensor<TF_FLOAT>>(tensor);

	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}

Tensor<TF_INT32> *make_int_tensor(const int32_t* array, int64_t len)
{
	Tensor<TF_INT32> tensor(array, len);
	auto tensor_ptr = std::make_shared<Tensor<TF_INT32>>(tensor);

	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}

float get_tensor1d_float_value_at(Tensor<TF_FLOAT> *tensor, int64_t idx)
{
	return get_tensor_float_value_at(tensor, &idx, 1);
}

float get_tensor_float_value_at(Tensor<TF_FLOAT> *tensor, int64_t *idxs, size_t len)
{
	return LifetimeManager::instance().accessOwned(tensor)->at(idxs, len);
}

int get_tensor1d_int_value_at(Tensor<TF_INT32> *tensor, int64_t idx)
{
	return get_tensor_int_value_at(tensor, &idx, 1);
}

int get_tensor_int_value_at(Tensor<TF_INT32> *tensor, int64_t *idxs, size_t len)
{
	return LifetimeManager::instance().accessOwned(tensor)->at(idxs, len);
}

