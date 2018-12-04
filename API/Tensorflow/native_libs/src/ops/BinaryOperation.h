#ifndef TFL_BINARYOPERATION_H
#define TFL_BINARYOPERATION_H

#include <string>
#include <memory>
#include <utility>
#include <tensorflow/c/c_api.h>
#include "Operation.h"
#include "../helpers/utils.h"
#include "../tensor/Tensor.h"

template<TF_DataType DataTypeLabel>
class BinaryOperation : public Operation<DataTypeLabel> {
public:
    BinaryOperation(std::string name,
                    std::shared_ptr<Operation<DataTypeLabel>> a,
                    std::shared_ptr<Operation<DataTypeLabel>> b)
        : operation_name(std::move(name)), arg1(std::move(a)), arg2(std::move(b)) {
        hash = std::hash<std::string>()(operation_name);
        hash = hash_combine(hash, arg1->hashcode());
        hash = hash_combine(hash, arg2->hashcode());
    }

    size_t hashcode() const override {
        return hash;
    }

    TF_Output addToGraph(TF_Graph* graph) const override {
        TF_OperationDescription *desc = TF_NewOperation(graph, operation_name.c_str(), std::to_string(hashcode()).c_str());

        TF_Output output1 = arg1->addToGraph(graph);
        TF_Output output2 = arg2->addToGraph(graph);

        TF_AddInput(desc, output1);
        TF_AddInput(desc, output2);

        TF_Operation *operation = run_with_status<TF_Operation*>(std::bind(TF_FinishOperation, desc, std::placeholders::_1));

        return {
                .oper = operation,
                .index = 0
        };
    };
protected:
    size_t hash;
    std::string operation_name;
    std::shared_ptr<Operation<DataTypeLabel>> arg1, arg2;
};

template<TF_DataType DataTypeLabel>
class Gradient : public BinaryOperation<DataTypeLabel> {
public:
    Gradient(std::shared_ptr<Operation<DataTypeLabel>> a,
                    std::shared_ptr<Operation<DataTypeLabel>> b)
        : BinaryOperation<DataTypeLabel>("Gradient", a, b) {}


    TF_Output addToGraph(TF_Graph* graph) const override {
        TF_Output output1 = this->arg1->addToGraph(graph);
        TF_Output output2 = this->arg2->addToGraph(graph);


		TF_Output output;
		run_with_status<void>(std::bind(TF_AddGradients, graph, &output1, 1, &output2, 1, nullptr, std::placeholders::_1, &output));

		return output;
    };
};

#endif //TFL_BINARYOPERATION_H
