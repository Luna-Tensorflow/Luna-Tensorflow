#ifndef FFITESTHELPER_CONST_H
#define FFITESTHELPER_CONST_H


#include <API/native_libs/src/utils.h>
#include <memory>
#include <utility>
#include <API/native_libs/src/tensor/Tensor.h>
#include "Operation.h"

template <TF_DataType DataTypeLabel>
class Const : public Operation {
public:
    explicit Const(std::shared_ptr<Tensor<DataTypeLabel>> tensor) : value(std::move(tensor)) {
        hash = value->hash();
    }

    size_t hashcode() override {
        return hash;
    }
private:
    size_t hash;
    std::shared_ptr<Tensor<DataTypeLabel>> value;
};

#endif //FFITESTHELPER_CONST_H
