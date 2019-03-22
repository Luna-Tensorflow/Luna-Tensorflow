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

TFL_API Tensor *make_tensor(void const *array, TF_DataType type, const int64_t *dims, size_t num_dims, const char **outError);
TFL_API int get_tensor_num_dims(Tensor *tensor, const char **outError);
TFL_API int64_t get_tensor_dim(Tensor *tensor, int32_t dim_index, const char **outError);
TFL_API int64_t* get_tensor_dims(Tensor *tensor, const char **outError);
TFL_API int64_t get_tensor_flatlist_length(Tensor* tensor, const char **outError);

#define DEFINE_TENSOR(typelabel) \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_##typelabel(Tensor *tensor, int64_t *idxs, size_t idxs_len, const char **outError); \
TFL_API Type<typelabel>::lunatype get_tensor_value_at_index_##typelabel(Tensor *tensor, int64_t index, const char **outError); \
TFL_API Tensor *make_random_tensor_##typelabel(const int64_t *dims, size_t num_dims, \
	Type<typelabel>::lunatype const min, Type<typelabel>::lunatype const max, const char **outError); \
TFL_API Tensor *make_const_tensor_##typelabel(const int64_t *dims, size_t num_dims, \
	Type<typelabel>::lunatype const value, const char **outError); \
TFL_API Type<typelabel>::lunatype* tensor_to_flatlist_##typelabel(Tensor*, const char **outError);


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

#ifdef __cplusplus
};
#endif

#endif //TFL_TENSOR_WRAPPER_H
