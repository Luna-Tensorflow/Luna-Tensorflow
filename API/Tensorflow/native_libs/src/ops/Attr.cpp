//
// Created by wojtek on 13.02.19.
//

#include <functional>
#include "Attr.h"
#include "../helpers/utils.h"

using namespace std;

Attr::Attr(const string &name) : name(name) {
}

AttrType::AttrType(const string &name, TF_DataType type) : Attr(name), type(type) {
}

void AttrType::set(TF_OperationDescription *desc) const {
    TF_SetAttrType(desc, name.c_str(), type);
}

AttrShape::AttrShape(const string &name, const vector<int64_t> &dims) : Attr(name), dims(dims) {
}

void AttrShape::set(TF_OperationDescription *desc) const {
    TF_SetAttrShape(desc, name.c_str(), dims.data(), dims.size());
}

AttrTensor::AttrTensor(const std::string &name, const Tensor &tensor) : Attr(name), tensor(tensor) {
}

void AttrTensor::set(TF_OperationDescription *desc) const {
    run_with_status<void>(std::bind(TF_SetAttrTensor, desc, name.c_str(), tensor.get_underlying(), std::placeholders::_1));
}