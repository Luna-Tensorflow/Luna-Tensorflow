//
// Created by wojtek on 13.02.19.
//

#include <functional>
#include <algorithm>
#include "Attr.h"
#include "../helpers/utils.h"

Attr::Attr(const std::string &name) : name(name) {
    static size_t hash_ctr = 0;
    hash = ++hash_ctr;
}

size_t Attr::hashcode() const {
    return hash;
}

AttrType::AttrType(const std::string &name, TF_DataType type) : Attr(name), type(type) {
}

void AttrType::set(TF_OperationDescription *desc) const {
    TF_SetAttrType(desc, name.c_str(), type);
}

AttrTypeList::AttrTypeList(const std::string &name, const std::vector<TF_DataType> &types) : Attr(name), types(types) {
}

void AttrTypeList::set(TF_OperationDescription *desc) const {
    TF_SetAttrTypeList(desc, name.c_str(), types.data(), types.size());
}

AttrShape::AttrShape(const std::string &name, const std::vector<int64_t> &dims) : Attr(name), dims(dims) {
}

void AttrShape::set(TF_OperationDescription *desc) const {
    TF_SetAttrShape(desc, name.c_str(), dims.data(), dims.size());
}

AttrShapeList::AttrShapeList(const std::string &name, std::vector<std::vector<int64_t>> dims) : Attr(name), dims(std::move(dims)) {
}

void AttrShapeList::set(TF_OperationDescription *desc) const {
    std::vector<const int64_t*> dims_pointers;
    std::transform(dims.begin(), dims.end(), dims_pointers.begin(),
            [](const std::vector<int64_t> &v){ return v.data(); });

    std::vector<int> dims_sizes;
    std::transform(dims.begin(), dims.end(), dims_sizes.begin(),
            [](const std::vector<int64_t> &v){ return v.size(); });

    TF_SetAttrShapeList(desc, name.c_str(), dims_pointers.data(), dims_sizes.data(), dims.size());
}

AttrTensor::AttrTensor(const std::string &name, const Tensor &tensor) : Attr(name), tensor(tensor) {
}

void AttrTensor::set(TF_OperationDescription *desc) const {
    run_with_status<void>(std::bind(TF_SetAttrTensor, desc, name.c_str(), tensor.get_underlying(),
            std::placeholders::_1));
}

AttrTensorList::AttrTensorList(const std::string &name, const std::vector<Tensor> &tensors) : Attr(name),
        tensors(tensors) {
}

void AttrTensorList::set(TF_OperationDescription *desc) const {
    std::vector<TF_Tensor*> tensor_pointers;
    std::transform(tensors.begin(), tensors.end(), tensor_pointers.begin(),
            [](const Tensor &tensor){ return tensor.get_underlying(); });

    run_with_status<void>(std::bind(TF_SetAttrTensorList, desc, name.c_str(), tensor_pointers.data(), tensors.size(),
            std::placeholders::_1));
}

AttrInt::AttrInt(const std::string &name, int64_t value) : Attr(name), value(value) {
}

void AttrInt::set(TF_OperationDescription *desc) const {
    TF_SetAttrInt(desc, name.c_str(), value);
}

AttrIntList::AttrIntList(const std::string &name, const std::vector<int64_t> &values) : Attr(name), values(values) {
}

void AttrIntList::set(TF_OperationDescription *desc) const {
    TF_SetAttrIntList(desc, name.c_str(), values.data(), values.size());
}

AttrFloat::AttrFloat(const std::string &name, float value) : Attr(name), value(value) {
}

void AttrFloat::set(TF_OperationDescription *desc) const {
    TF_SetAttrFloat(desc, name.c_str(), value);
}

AttrFloatList::AttrFloatList(const std::string &name, const std::vector<float> &values) : Attr(name), values(values) {
}

void AttrFloatList::set(TF_OperationDescription *desc) const {
    TF_SetAttrFloatList(desc, name.c_str(), values.data(), values.size());
}

AttrBool::AttrBool(const std::string &name, bool value) : Attr(name), value(value) {
}

void AttrBool::set(TF_OperationDescription *desc) const {
    unsigned char value_char = 0;
    if (value) {
        value_char = 1;
    }
    TF_SetAttrBool(desc, name.c_str(), value_char);
}

AttrBoolList::AttrBoolList(const std::string &name, const std::vector<bool> &values) : Attr(name), values(values) {
}

void AttrBoolList::set(TF_OperationDescription *desc) const {
    std::vector<unsigned char> uchar_values;
    std::transform(values.begin(), values.end(), uchar_values.begin(), [](bool v){ if (v) return 1; return 0; });
    TF_SetAttrBoolList(desc, name.c_str(), uchar_values.data(), uchar_values.size());
}

AttrString::AttrString(const std::string &name, const std::string &value) : Attr(name), value(value) {
}

void AttrString::set(TF_OperationDescription *desc) const {
    TF_SetAttrString(desc, name.c_str(), value.c_str(), value.size());
}

AttrStringList::AttrStringList(const std::string &name, const std::vector<std::string> &values) : Attr(name), values(values) {
}

void AttrStringList::set(TF_OperationDescription *desc) const {
    std::vector<const void*> values_pointers;
    std::transform(values.begin(), values.end(), values_pointers.begin(), [](const std::string& str){ return str.c_str(); });

    std::vector<size_t> values_sizes;
    std::transform(values.begin(), values.end(), values_sizes.begin(), [](const std::string& str){ return str.size(); });

    TF_SetAttrStringList(desc, name.c_str(), values_pointers.data(), values_sizes.data(), values.size());
}

AttrFuncName::AttrFuncName(const std::string &name, const std::string &func_name) : Attr(name), func_name(func_name) {
}

void AttrFuncName::set(TF_OperationDescription *desc) const {
    TF_SetAttrFuncName(desc, name.c_str(), func_name.c_str(), func_name.size());
}
