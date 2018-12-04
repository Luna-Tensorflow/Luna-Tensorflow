#ifndef TFL_CONST_H
#define TFL_CONST_H


#include <memory>
#include <string>
#include <utility>
#include <functional>
#include <tensorflow/c/c_api.h>
#include "../tensor/Tensor.h"
#include "../helpers/utils.h"
#include "Operation.h"

template <TF_DataType DataTypeLabel>
class Const : public Operation<DataTypeLabel> {
public:
    explicit Const(std::shared_ptr<Tensor<DataTypeLabel>> tensor) : value(std::move(tensor)) {
        hash = value->hash();
    }

    size_t hashcode() const override {
        return hash;
    }

    TF_Output addToGraph(TF_Graph* graph) const override {
        TF_OperationDescription *desc = TF_NewOperation(graph, "Const", std::to_string(hashcode()).c_str());

        TF_SetAttrType(desc, "dtype", DataTypeLabel);
        run_with_status<void>(std::bind(TF_SetAttrTensor, desc, "value", value->get_underlying(), std::placeholders::_1));

        TF_Operation *operation = run_with_status<TF_Operation*>(std::bind(TF_FinishOperation, desc, std::placeholders::_1));

        return {
            .oper = operation,
            .index = 0
        };
    }

    // std::vector<int64_t> getShape(){
    //     return value.shape();
    // }

private:
    size_t hash;
    std::shared_ptr<Tensor<DataTypeLabel>> value;
};

#endif //TFL_CONST_H
