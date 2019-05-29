#ifndef TFL_ATTRIBUTES_H
#define TFL_ATTRIBUTES_H

#include <vector>
#include <memory>
#include <cstddef>

#include "tensorflow/c/c_api.h"

#include "common.h"
#include "../ops/Attr.h"

#ifdef __cplusplus
extern "C"
{
#endif

TFL_API std::vector<std::shared_ptr<Attr>>* attr_list_init(const char **outError);

TFL_API void add_attr_type(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, TF_DataType type, const char **outError);
TFL_API void add_attr_type_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, TF_DataType* types, uint32_t num_types, const char **outError);

TFL_API void add_attr_shape(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t* dims, uint32_t num_dims, const char **outError);
TFL_API void add_attr_shape_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t** dims, uint32_t* dims_sizes, uint32_t num_dims, const char **outError);

TFL_API void add_attr_tensor(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, Tensor* tensor, const char **outError);
TFL_API void add_attr_tensor_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, Tensor** tensors, uint32_t num_tensors, const char **outError);

TFL_API void add_attr_int(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t value, const char **outError);
TFL_API void add_attr_int_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t* values, uint32_t num_values, const char **outError);

TFL_API void add_attr_float(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, float value, const char **outError);
TFL_API void add_attr_float_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, float* values, uint32_t num_values, const char **outError);

TFL_API void add_attr_bool(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, unsigned char value, const char **outError);
TFL_API void add_attr_bool_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, unsigned char* values, uint32_t num_values, const char **outError);

TFL_API void add_attr_string(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, char* value, const char **outError);
TFL_API void add_attr_string_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, char** values, uint32_t num_values, const char **outError);

TFL_API void add_attr_func_name(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, char* func_name, const char **outError);

#ifdef __cplusplus
}
#endif

#endif //TFL_ATTRIBUTES_H
