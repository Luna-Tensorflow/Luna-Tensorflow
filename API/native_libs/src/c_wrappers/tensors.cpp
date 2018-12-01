//
// Created by mateusz on 01.12.18.
//

#include "tensors.h"
#include <API/native_libs/src/tensor/Tensor.h>
#include <API/native_libs/src/LifeTimeManager.h>

#include <vector>
#include <memory>

Tensor<TF_FLOAT> *make_float_tensor(const float* array, size_t len)
{
	Tensor<TF_FLOAT> tensor(array, len);
	auto tensor_ptr = std::make_shared<Tensor<TF_FLOAT>>(tensor);

	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}

Tensor<TF_FLOAT> *make_float_tensor_arr(const float** array, size_t width, size_t height)
{
	Tensor<TF_FLOAT> tensor(array, width, height);
	auto tensor_ptr = std::make_shared<Tensor<TF_FLOAT>>(tensor);

	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}

Tensor<TF_INT32> *make_int_tensor(const int32_t* array, size_t len)
{
	Tensor<TF_INT32> tensor(array, len);
	auto tensor_ptr = std::make_shared<Tensor<TF_INT32>>(tensor);

	return LifetimeManager::instance().addOwnership(std::move(tensor_ptr));
}
