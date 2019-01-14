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
TFL_API Tensor<typelabel> *make_tensor_##typelabel(Type<typelabel>::lunatype const *array, const int64_t *dims, size_t num_dims); \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_##typelabel(Tensor<typelabel> *tensor, int64_t *idxs, size_t idxs_len); \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_index_##typelabel(Tensor<typelabel> *tensor, int64_t index); \
TFL_API int get_tensor_num_dims_##typelabel(Tensor<typelabel> *tensor); \
TFL_API int64_t get_tensor_dim_##typelabel(Tensor<typelabel> *tensor, int32_t dim_index); \


DEFINE_TENSOR(TF_FLOAT);
DEFINE_TENSOR(TF_DOUBLE);
DEFINE_TENSOR(TF_INT8);
DEFINE_TENSOR(TF_INT16);
DEFINE_TENSOR(TF_INT32);
DEFINE_TENSOR(TF_INT64);
DEFINE_TENSOR(TF_UINT8);
DEFINE_TENSOR(TF_UINT16);
DEFINE_TENSOR(TF_UINT32);
DEFINE_TENSOR(TF_UINT64);
DEFINE_TENSOR(TF_BOOL);
DEFINE_TENSOR(TF_STRING);
//DEFINE_TENSOR(TF_HALF);

// TODO remove old definitions after migration is complete
TFL_API Tensor<TF_FLOAT> *make_float_tensor(float const*, int64_t);
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
