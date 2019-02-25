#include "attributes.h"
#include "../helpers/LifeTimeManager.h"
#include "../helpers/logging.h"

std::vector<std::shared_ptr<Attr>>* attr_list_init() {
    LOG_PARAMLESS;
    auto attr_list_ptr = std::make_shared<std::vector<std::shared_ptr<Attr>>>();
    return LifetimeManager::instance().addOwnership(std::move(attr_list_ptr));
}

void add_attr_type(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, TF_DataType type) {
    LOG(attr_list, name, type);
    LifetimeManager::instance().accessOwned(attr_list)->push_back(std::make_shared<AttrType>(std::string(name), type));
}

void add_attr_shape(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, int64_t* dims, size_t num_dims) {
    LOG(attr_list, name, dims, num_dims);
    LifetimeManager::instance().accessOwned(attr_list)->push_back(std::make_shared<AttrShape>(std::string(name), std::vector<int64_t>(dims, dims + num_dims)));
}

void add_attr_tensor(std::vector<std::shared_ptr<Attr>>* attr_list, char* name, Tensor* tensor) {
    LOG(attr_list, name, tensor);
    LifetimeManager::instance().accessOwned(attr_list)->push_back(std::make_shared<AttrTensor>(std::string(name), *LifetimeManager::instance().accessOwned(tensor)));
}

void add_attr_int(std::vector<std::shared_ptr<Attr>> *attr_list, char *name, int64_t value) {
    LOG(attr_list, name, value);
    LifetimeManager::instance().accessOwned(attr_list)->push_back(std::make_shared<AttrInt>(std::string(name), value));
}

void add_attr_float(std::vector<std::shared_ptr<Attr>> *attr_list, char *name, float value) {
    LOG(attr_list, name, value);
    LifetimeManager::instance().accessOwned(attr_list)->push_back(std::make_shared<AttrFloat>(std::string(name), value));
}

void add_attr_bool(std::vector<std::shared_ptr<Attr>> *attr_list, char *name, unsigned char value) {
    LOG(attr_list, name, value);
    bool value_bool = 0;
    if (value) {
        value_bool = 1;
    }
    LifetimeManager::instance().accessOwned(attr_list)->push_back(std::make_shared<AttrBool>(std::string(name), value_bool));
}

void add_attr_string(std::vector<std::shared_ptr<Attr>> *attr_list, char *name, char *value) {
    LOG(attr_list, name, value);
    LifetimeManager::instance().accessOwned(attr_list)->push_back(std::make_shared<AttrString>(std::string(name), std::string(value)));
}
