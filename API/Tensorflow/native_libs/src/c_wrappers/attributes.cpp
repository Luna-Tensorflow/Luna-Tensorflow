#include <algorithm>

#include "attributes.h"
#include "../helpers/LifeTimeManager.h"
#include "../helpers/logging.h"
#include "../helpers/error.h"

std::vector<std::shared_ptr<Attr>>* attr_list_init(const char **outError) {
    return TRANSLATE_EXCEPTION(outError) {
        FFILOG_PARAMLESS;
        auto attr_list_ptr = std::make_shared<std::vector<std::shared_ptr<Attr>>>();
        return LifetimeManager::instance().addOwnership(std::move(attr_list_ptr));
    };
}

void add_attr_type(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, TF_DataType type, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, type);
        attr_list->push_back(std::make_shared<AttrType>(std::string(name), type));
    };
}

void add_attr_type_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, TF_DataType* types, uint32_t num_types, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, types, num_types);
        auto new_attr = std::make_shared<AttrTypeList>(std::string(name), std::vector<TF_DataType>(types, types + num_types));
        attr_list->push_back(std::move(new_attr));
    };
}

void add_attr_shape(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t* dims, uint32_t num_dims, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, dims, num_dims);
        attr_list->push_back(std::make_shared<AttrShape>(std::string(name), std::vector<int64_t>(dims, dims + num_dims)));
    };
}

void add_attr_shape_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t** dims, uint32_t* dims_sizes, uint32_t num_dims, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, dims, dims_sizes, num_dims);
        std::vector<std::vector<int64_t>> dims_v;
        for (uint32_t i = 0; i < num_dims; ++i) {
            dims_v.emplace_back(dims[i], dims[i] + dims_sizes[i]);
        }
        attr_list->push_back(std::make_shared<AttrShapeList>(std::string(name), std::move(dims_v)));
    };
}

void add_attr_tensor(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, Tensor* tensor, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, tensor);
        attr_list->push_back(std::make_shared<AttrTensor>(std::string(name), *tensor));
    };
}

void add_attr_tensor_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, Tensor** tensors, uint32_t num_tensors, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, tensors, num_tensors);
        std::vector<Tensor> tensors_v;
        for (uint32_t i = 0; i < num_tensors; ++i) {
            tensors_v.push_back(*tensors[i]);
        }
        attr_list->push_back(std::make_shared<AttrTensorList>(std::string(name), std::move(tensors_v)));
    };
}

void add_attr_int(std::vector<std::shared_ptr<Attr>> *attr_list, char *name, int64_t value, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, value);
        attr_list->push_back(std::make_shared<AttrInt>(std::string(name), value));
    };
}

void add_attr_int_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t* values, uint32_t num_values, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, values, num_values);
        attr_list->push_back(std::make_shared<AttrIntList>(std::string(name), std::vector<int64_t>(values, values + num_values)));
    };
}

void add_attr_float(std::vector<std::shared_ptr<Attr>> *attr_list, char *name, float value, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, value);
        attr_list->push_back(std::make_shared<AttrFloat>(std::string(name), value));
    };
}

void add_attr_float_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, float* values, uint32_t num_values, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, values, num_values);
        attr_list->push_back(std::make_shared<AttrFloatList>(std::string(name), std::vector<float>(values, values + num_values)));
    };
}

void add_attr_bool(std::vector<std::shared_ptr<Attr>> *attr_list, char *name, unsigned char value, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, value);
        bool value_bool = 0;
        if (value) {
            value_bool = 1;
        }
        attr_list->push_back(std::make_shared<AttrBool>(std::string(name), value_bool));
    };
}

void add_attr_bool_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, unsigned char* values, uint32_t num_values, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, values, num_values);
        attr_list->push_back(std::make_shared<AttrBoolList>(std::string(name), std::vector<bool>(values, values + num_values)));
    };
}

void add_attr_string(std::vector<std::shared_ptr<Attr>> *attr_list, char *name, char *value, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, value);
        attr_list->push_back(std::make_shared<AttrString>(std::string(name), std::string(value)));
    };
}

void add_attr_string_list(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, char** values, uint32_t num_values, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, values, num_values);
        attr_list->push_back(std::make_shared<AttrStringList>(std::string(name), std::vector<std::string>(values, values + num_values)));
    };
}

void add_attr_func_name(std::vector<std::shared_ptr<Attr>> *attr_list, char *name, char *func_name, const char **outError) {
    TRANSLATE_EXCEPTION(outError) {
        FFILOG(attr_list, name, func_name);
        attr_list->push_back(std::make_shared<AttrFuncName>(std::string(name), std::string(func_name)));
    };
}
