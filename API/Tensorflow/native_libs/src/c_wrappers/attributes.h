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

TFL_API std::vector<std::shared_ptr<Attr>>* attr_list_init();
TFL_API void add_attr_type(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, TF_DataType type);
TFL_API void add_attr_shape(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t* dims, size_t num_dims);
TFL_API void add_attr_tensor(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, Tensor* tensor);
TFL_API void add_attr_int(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t value);
TFL_API void add_attr_float(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, float value);
TFL_API void add_attr_bool(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, unsigned char value);
TFL_API void add_attr_string(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, char* value);

#ifdef __cplusplus
};
#endif

#endif //TFL_ATTRIBUTES_H
