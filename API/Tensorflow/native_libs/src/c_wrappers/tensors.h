//
// Created by mateusz on 01.12.18.
//

#ifndef TFL_TENSOR_WRAPPER_H
#define TFL_TENSOR_WRAPPER_H

#include "../tensor/Tensor.h"
#include "../tensor/TypeLabel.h"
#include "common.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define DEFINE_TENSOR(typelabel) \
TFL_API Tensor<typelabel> *make_tensor_##typelabel(Type<typelabel>::type const *array, int64_t len); \
TFL_API Tensor<typelabel> *make_tensor_arr_##typelabel(Type<typelabel>::type const **array, int64_t width, int64_t height); \
TFL_API Type<typelabel>::type get_tensor_value_at_##typelabel(Tensor<typelabel> *tensor, int64_t *idxs, size_t idxs_len); \
TFL_API int64_t get_tensor_length_##typelabel(Tensor<typelabel> *tensor);


DEFINE_TENSOR(TF_FLOAT);
DEFINE_TENSOR(TF_INT32);

// TODO remove old definitions after migration is complete
TFL_API Tensor<TF_FLOAT> *make_float_tensor(float const*, int64_t);
TFL_API Tensor<TF_FLOAT> *make_float_tensor_arr(float const**, int64_t, int64_t);
TFL_API Tensor<TF_INT32> *make_int_tensor(int32_t const*, int64_t);

TFL_API float get_tensor1d_float_value_at(Tensor<TF_FLOAT> *, int64_t);
TFL_API float get_tensor_float_value_at(Tensor<TF_FLOAT> *, int64_t*, size_t);

TFL_API int32_t get_tensor1d_int_value_at(Tensor<TF_INT32> *, int64_t);
TFL_API int32_t get_tensor_int_value_at(Tensor<TF_INT32> *, int64_t*, size_t);

TFL_API int64_t tensor_float_length(Tensor<TF_FLOAT> *);

#ifdef __cplusplus
};
#endif

#endif //TFL_TENSOR_WRAPPER_H
