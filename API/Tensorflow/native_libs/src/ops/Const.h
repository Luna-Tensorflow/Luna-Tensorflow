#ifndef TFL_CONST_H
#define TFL_CONST_H


#include <memory>
#include <utility>
#include "../tensor/Tensor.h"
#include "../utils.h"
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
