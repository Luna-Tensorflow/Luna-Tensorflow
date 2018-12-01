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
	Tensor<TF_FLOAT> *make_float_tensor(const float*, int);
	Tensor<TF_FLOAT> *make_float_tensor_arr(const float**, int, int);
	Tensor<TF_INT32> *make_int_tensor(const int32_t*, int);


#ifdef __cplusplus
};
#endif

#endif //TFL_TENSOR_WRAPPER_H
