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


#ifdef __cplusplus
};
#endif

#endif //TFL_TENSOR_WRAPPER_H
