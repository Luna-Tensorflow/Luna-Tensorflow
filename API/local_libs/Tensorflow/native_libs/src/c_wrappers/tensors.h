//
// Created by mateusz on 01.12.18.
//

#ifndef TFL_TENSOR_WRAPPER_H
#define TFL_TENSOR_WRAPPER_H

#include <API/native_libs/src/tensor/Tensor.h>

#ifdef __cplusplus
extern "C"
{
#endif
	Tensor<TF_FLOAT> *make_float_tensor(float const*, int64_t);
	Tensor<TF_FLOAT> *make_float_tensor_arr(float const**, int64_t, int64_t);
	Tensor<TF_INT32> *make_int_tensor(int32_t const*, int64_t);

	float get_tensor1d_float_value_at(Tensor<TF_FLOAT> *, int64_t);
	float get_tensor_float_value_at(Tensor<TF_FLOAT> *, int64_t*, size_t);

	int32_t get_tensor1d_int_value_at(Tensor<TF_INT32> *, int64_t);
   int32_t get_tensor_int_value_at(Tensor<TF_INT32> *, int64_t*, size_t);

    int64_t tensor_float_length(Tensor<TF_FLOAT> *);

#ifdef __cplusplus
};
#endif

#endif //TFL_TENSOR_WRAPPER_H
